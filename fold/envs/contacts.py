import torch
import genesis as gs
import seaborn as sns

def get_contact_marker_cfgs(
    num_vis_contacts=10,
    radius=0.008,
    sources=['demo', 'policy'],
    obj_parts=['top', 'bottom'],
    hand_sides=['left', 'right'],
):
    SNS_CUBE = sns.color_palette('cubehelix', 8)
    SNS_SUMMER = sns.color_palette("summer", 4)
    SNS_SPRING = sns.color_palette("spring", 4)
    SNS_RED = sns.color_palette("Reds", 4)
    SNS_BLUE = sns.color_palette("Blues", 4)
    CONTACT_MARKER_COLORS = {
        name: SNS_SUMMER[i] for i, name in enumerate([
            'demo_top_left', 'demo_top_right', 'demo_bottom_left', 'demo_bottom_right',
        ])
    }
    CONTACT_MARKER_COLORS.update({
        name: SNS_BLUE[-1] for i, name in enumerate([
            'policy_top_left', 'policy_top_right', 'policy_bottom_left', 'policy_bottom_right',
        ])
    })
    # generates all combos of sources, obj_parts, hand_sides
    contact_marker_cfgs = dict()
    for source in sources:
        for part in obj_parts:
            for side in hand_sides:
                name = f"{source}_{part}_{side}"
                color = CONTACT_MARKER_COLORS[name]
                contact_marker_cfgs[name] = {
                    'num_vis_contacts': num_vis_contacts,
                    'color': color + (0.8,),
                    'radius': radius,
                }
    return contact_marker_cfgs


"""
Note that might need to specify options.max_collision_pairs to get more contacts.
see gs.options.solvers.RigidOptions(max_collision_pairs=100) default is 100
"""
@torch.jit.script
def get_expanded_mask(n_contacts, pair_a, pair_b, a_idxs, b_idxs):
    """
    Input:
    - n_contacts: (N,) number of contacts, must use this to consider only top k contacts
    - pair_a: (n_max_contacts, N) tensor of geom indices for entity A
    - pair_b: (n_max_contacts, N) tensor of geom indices for entity B
    - a_idxs: (n_a,) tensor of geom indices to filter for entity A
    - b_idxs: (n_b,) tensor of geom indices to filter for entity B
    Output:
    - mask: (n_max_contacts, N, n_a, n_b) tensor of expanded mask. Use this to filter contacts

    Note that we need to flip the mask for both pairs
    """
    # set out-of-bound indices to -3
    pair_a[n_contacts.unsqueeze(0) <= torch.arange(pair_a.shape[0], device=pair_a.device).view(-1, 1)] = -3
    pair_b[n_contacts.unsqueeze(0) <= torch.arange(pair_b.shape[0], device=pair_b.device).view(-1, 1)] = -3
    assert a_idxs.ndim == 1 and b_idxs.ndim == 1, "a_idxs and b_idxs should be 1D tensors"
    mask_a = pair_a.unsqueeze(-1) == a_idxs.view(1, 1, -1)
    mask_b = pair_b.unsqueeze(-1) == b_idxs.view(1, 1, -1)
    mask = torch.logical_and(mask_a.unsqueeze(-1), mask_b.unsqueeze(-2))
    mask_a_flip = pair_a.unsqueeze(-1) == b_idxs.view(1, 1, -1)
    mask_b_flip = pair_b.unsqueeze(-1) == a_idxs.view(1, 1, -1)
    mask_flip = torch.logical_and(mask_a_flip.unsqueeze(-1), mask_b_flip.unsqueeze(-2))

    mask = torch.logical_or(mask, mask_flip.transpose(-1, -2))

    return mask

@torch.jit.script
def get_filtered_contact_force_from_mask(force, mask):
    """
    Input:
    - force: (n_contacts, N, 3) tensor of contact forces
    - mask: (n_contacts, N, n_a, n_b) tensor of expanded mask, n_a is either num of geoms or num of links
    Return:
    - summed force tensor of shape (N, n_a, n_b, 3) """
    mask_expand = mask.unsqueeze(-1)
    force_expand = force.unsqueeze(-2).unsqueeze(-2)
    sum_force = (force_expand * mask_expand).sum(dim=0)
    return sum_force


# @torch.jit.script
def get_all_contact_pos_from_mask(contact_pos, mask):
    """
    Get the contact positions from the mask, this gets all the positions for each geom/link in entity A
    Input:
    - contact_pos: (n_contacts, N, 3) tensor of contact positions
    - contact_force_norm: (n_contacts, N) tensor of contact force norms
    - mask: (n_contacts, N, n_a, n_b) tensor of expanded mask
    Return:
    - all_contact_pos: (N, n_a, n_contacts, 3) tensor of contact positions
    - all_contact_pos_valid: (N, n_a, n_contacts) tensor of valid contact positions
    """
    device = contact_pos.device
    n_a = mask.shape[-2]
    n_b = mask.shape[-1]
    n_con = contact_pos.shape[0]
    n_envs = contact_pos.shape[1]

    all_contact_pos = torch.zeros(n_con, n_envs, n_a, 3).to(device) # allocate max n_con contacts for each geom/link in entity A
    all_contact_pos_valid = torch.zeros((n_con, n_envs, n_a), dtype=torch.bool).to(device) # mask for valid contact positions

    for idx in range(n_a):
        coll_b_msk = mask[:, :, idx, :] # shape (n_contacts, N, n_b)
        any_b_msk = coll_b_msk.any(dim=-1) # shape (n_contacts, N)
        all_contact_pos[:, :, idx] = contact_pos * any_b_msk.unsqueeze(-1)
        all_contact_pos_valid[:, :, idx] = any_b_msk
    # breakpoint()
    # reshape
    all_contact_pos = all_contact_pos.permute(1, 2, 0, 3) # shape (N, n_a, n_contacts, 3)
    all_contact_pos_valid = all_contact_pos_valid.permute(1, 2, 0) # shape (N, n_a, n_contacts)
    return all_contact_pos, all_contact_pos_valid

@torch.jit.script
def get_grouped_contact_pos(contact_pos, contact_force_norm, mask):
    """
    Get the contact positions from the mask, this groups the contact positions by geom/link in entity A and all the valid contacts with entity B
    Input:
    - contact_pos: (n_contacts, N, 3) tensor of contact positions
    - contact_force_norm: (n_contacts, N) tensor of contact force norms
    - mask: (n_contacts, N, n_a, n_b) tensor of expanded mask
    Return:
    - grouped_contact_pos: (N, n_a, n_b, 3) tensor of contact positions
    - grouped_contact_pos_valid: (N, n_a, n_b) tensor of valid contact positions
    """
    device = contact_pos.device
    contact_force_norm = contact_force_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (n_contacts, N, 1, 1, 1)
    pos_expanded = contact_pos.unsqueeze(-2).unsqueeze(-2) # (n_contacts, N, 1, 1, 3)
    pos_masked = pos_expanded * mask.unsqueeze(-1)  # Mask invalid positions -> (n_contacts, N, n_a, n_b, 3)
    weightd_pos = pos_expanded * contact_force_norm  # (n_contacts, N, 1, 1, 3)
    weighted_pos_masked = weightd_pos * mask.unsqueeze(-1)  # Mask invalid positions -> (n_contacts, N, n_a, n_b, 3)
    # Sum weighted positions and contact forces across all contacts (dim=0)
    weighted_sum = torch.sum(weighted_pos_masked, dim=0)  # (N, n_a, n_b, 3)
    force_sum = torch.sum(contact_force_norm * mask.unsqueeze(-1), dim=0)  # (N, n_a, n_b, 1)
    # Compute the weighted average (avoid division by zero)
    grouped_contact_pos = torch.where(
        force_sum > 0,
        weighted_sum / force_sum,
        torch.zeros_like(weighted_sum)
    )
    # Compute validity (at least one valid contact)
    grouped_contact_pos_valid = (force_sum > 0).squeeze(-1) # (N, n_a, n_b)
    return grouped_contact_pos, grouped_contact_pos_valid

def get_filtered_contacts(
    entity_a,
    entity_b=None,
    filter_geoms_a=[],
    filter_geoms_b=[],
    filter_links_a=[],
    filter_links_b=[],
    return_geom_force=False,
    return_link_force=False,
    return_geom_pos=False,
    return_link_pos=False,
    device='cuda:0', # NOTE: to_torch() default device is cpu
):
    """
    Get the filtered contacts between two entities.
    Each entity has a number of geoms, if specified, only contacts between the specified geoms are returned.

    return:
    - contact_pos_geom_a: (N, n_geom_a, max_n, 3) -> up to max_n contacts between each geom in entity_a and other geoms of entity_b
    - contact_pos_geom_valid: (N, n_geom_a, max_n, ) -> mask for valid contact positions, because some pairs might not have contacts

    - contact_pos_link: (N, n_link_a, n_link_b, 3) -> Average all the found positions between link_a and link_b using the contact force norm
    - contact_pos_link_valid: (N, n_link_a, n_link_b) -> mask for valid contact positions

    - contact_force_a_geom: (N, n_geom_a, n_geom_b, 3) -> aggregated contact force on geom_a (flip sign for geom_b)
    - contact_force_a_link: (N, n_link_a, n_link_b, 3) -> aggregated contact force on link_a (flip sign for link_b)
    """
    collider = entity_a._solver.collider

    # Genesis API compatibility: old versions expose `collider.contact_data`,
    # newer versions keep it under `collider._collider_state.contact_data`.
    contact_data = getattr(collider, "contact_data", None)
    n_contacts_field = getattr(collider, "n_contacts", None)
    if contact_data is None or n_contacts_field is None:
        collider_state = getattr(collider, "_collider_state", None)
        if collider_state is not None:
            contact_data = getattr(collider_state, "contact_data", contact_data)
            n_contacts_field = getattr(collider_state, "n_contacts", n_contacts_field)

    if contact_data is not None and n_contacts_field is not None:
        n_contacts = n_contacts_field.to_torch(device=device)  # size [N]
        force = contact_data.force.to_torch(device=device)  # size [n_max, N, 3]
        geom_a_all = contact_data.geom_a.to_torch(device=device)
        geom_b_all = contact_data.geom_b.to_torch(device=device)
        link_a_all = contact_data.link_a.to_torch(device=device)
        link_b_all = contact_data.link_b.to_torch(device=device)
        pos_all = contact_data.pos.to_torch(device=device)
    else:
        # Public fallback path for newer collider implementations.
        contacts_info = collider.get_contacts(as_tensor=True, to_torch=True, keep_batch_dim=True)
        collider_state = getattr(collider, "_collider_state", None)
        if collider_state is None or not hasattr(collider_state, "n_contacts"):
            raise AttributeError("Collider does not expose contact tensors in a compatible format.")
        n_contacts = collider_state.n_contacts.to_torch(device=device)
        # get_contacts returns [N, n_max, ...]; convert to [n_max, N, ...]
        geom_a_all = contacts_info["geom_a"].transpose(0, 1).to(device)
        geom_b_all = contacts_info["geom_b"].transpose(0, 1).to(device)
        link_a_all = contacts_info["link_a"].transpose(0, 1).to(device)
        link_b_all = contacts_info["link_b"].transpose(0, 1).to(device)
        pos_all = contacts_info["position"].transpose(0, 1).to(device)
        force = contacts_info["force"].transpose(0, 1).to(device)
    contact_info = dict()

    if return_geom_force or return_geom_pos:
        geom_a = geom_a_all
        geom_b = geom_b_all
        if len(filter_geoms_a) == 0:
            geom_a_idxs = torch.tensor([i for i in range(entity_a.geom_start, entity_a.geom_end)]).to(force.device)
        else:
            geom_a_idxs = torch.tensor(filter_geoms_a).to(force.device)
        if len(filter_geoms_b) == 0:
            geom_b_idxs = torch.tensor([i for i in range(entity_b.geom_start, entity_b.geom_end)]).to(force.device)
        else:
            geom_b_idxs = torch.tensor(filter_geoms_b).to(force.device)
        geom_mask = get_expanded_mask(n_contacts, geom_a, geom_b, geom_a_idxs, geom_b_idxs)

    if return_link_force or return_link_pos:
        link_a = link_a_all
        link_b = link_b_all

        if len(filter_links_a) > 0:
            # assert is tensor:
            assert isinstance(filter_links_a, torch.Tensor), "filter_links_a should be a tensor"
            link_a_idxs = filter_links_a.to(force.device)
        else:
            link_a_idxs = torch.tensor([i for i in range(entity_a.link_start, entity_a.link_end)]).to(force.device)

        if len(filter_links_b) > 0:
            link_b_idxs = filter_links_b.to(force.device)
        else:
            assert entity_b is not None, "entity_b should be specified if filter_links_b is not"
            link_b_idxs = torch.tensor([i for i in range(entity_b.link_start, entity_b.link_end)]).to(force.device)
        link_mask = get_expanded_mask(n_contacts, link_a, link_b, link_a_idxs, link_b_idxs)

    if return_geom_force: # 500 FPS
        ### Get contact force by geom ###
        # force_a_geom = index_contact_force(
        #     force, geom_a, geom_b, geom_a_idxs, geom_b_idxs
        #     ) # shape (N, n_geom_a, n_geom_b, 3)
        force_a_geom = get_filtered_contact_force_from_mask(force, geom_mask)
        contact_info['contact_force_geom_a'] = force_a_geom

    if return_link_force:
        ### Get contact positions by link ###
        # force_a_link = index_contact_force(
        #     force, link_a, link_b, link_a_idxs, link_b_idxs
        #     ) # shape (N, n_link_a, n_link_b, 3)
        force_a_link = get_filtered_contact_force_from_mask(force, link_mask)
        contact_info['contact_force_link_a'] = force_a_link

    if return_geom_pos or return_link_pos:
         ### Get contact positions by geom ###
        contact_pos = pos_all # size [n_max, N, 3]


    if return_geom_pos: #
        # geom_a_pos, geom_a_valid = get_per_geom_a_contact_pos(contact_pos, force, geom_a, geom_b, geom_a_idxs, geom_b_idxs)
        geom_a_pos, geom_a_valid = get_all_contact_pos_from_mask(contact_pos, geom_mask)
        contact_info['contact_pos_geom_a'] = geom_a_pos
        contact_info['contact_pos_geom_a_valid'] = geom_a_valid

    if return_link_pos:
        force_norm = torch.norm(force, dim=-1)
        # link_pos, link_valid = get_per_link_a_contact_pos(contact_pos, force_norm, link_a, link_b, link_a_idxs, link_b_idxs)

        link_pos, link_valid = get_grouped_contact_pos(contact_pos, force_norm, link_mask)
        contact_info['contact_pos_link_a'] = link_pos
        contact_info['contact_pos_link_a_valid'] = link_valid


    return contact_info


def index_contact_force(n_contacts, force, geom_a, geom_b, geom_a_idxs, geom_b_idxs):
    """
    Input:
    - force: (n_contacts, N, 3) tensor of contact forces
    - geom_a: (n_contacts, N) tensor of geom indices for entity A
    - geom_b: (n_contacts, N) tensor of geom indices for entity B
    - geom_a_idxs: 1D list of geom indices to filter for entity A
    - geom_b_idxs: 1D list of geom indices to filter for entity B
    Return:
    - summed force tensor of shape (N, n_geom_a, n_geom_b, 3) """
    device = force.device
    mask = get_expanded_mask(n_contacts, geom_a, geom_b, geom_a_idxs, geom_b_idxs)
    # mask_expand = mask.unsqueeze(-1) # shape (n_contacts, N, n_a, n_b, 1)
    # force_expand = force.unsqueeze(-2).unsqueeze(-2) # shape (n_contacts, N, 1, 1, 3)
    # sum_force = (force_expand * mask_expand).sum(dim=0) # shape (N, n_a, n_b, 3)
    sum_force = get_filtered_contact_force_from_mask(force, mask)
    return sum_force


def get_per_link_a_contact_pos(n_contacts, contact_pos, contact_force_norm, link_a, link_b, link_a_idxs, link_b_idxs):
    """
    Get all the contact positions between each link_a and link_b
    Input:
    - contact_pos: (n_contacts, N, 3) tensor of contact positions
    - contact_force_norm: (n_contacts, N) tensor of contact force norms
    - link_a: (n_contacts, N) tensor of link indices for entity A
    - link_b: (n_contacts, N) tensor of link indices for entity B
    - link_a_idxs: 1D list of link indices to filter for entity A
    - link_b_idxs: 1D list of link indices to filter for entity B
    Return:
    - contact_pos_link: (N, n_link_a, n_link_b, 3) tensor of contact positions
    - contact_pos_link_valid: (N, n_link_a, n_link_b) tensor of valid contact positions
    """
    device = contact_pos.device
    mask = get_expanded_mask(n_contacts, link_a, link_b, link_a_idxs, link_b_idxs)
    assert len(contact_pos.shape) == 3, "contact_pos should have shape (n_contacts, N, 3)"
    n_envs = contact_pos.shape[1]
    n_contacts = contact_pos.shape[0]
    # because there's at most n_contacts across all geoms in each env, we allocate the max size for each geom_a and fill in geom_b contacts
    contact_force_norm = contact_force_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (n_contacts, N, 1, 1, 1)
    pos_expanded = contact_pos.unsqueeze(-2).unsqueeze(-2) # (n_contacts, N, 1, 1, 3)
    # Compute weighted contact positions

    pos_masked = pos_expanded * mask.unsqueeze(-1)  # Mask invalid positions -> (n_contacts, N, n_link_a, n_link_b, 3)

    weighted_contact_pos = pos_expanded * contact_force_norm  # (n_contacts, N, 1, 1, 3)
    weighted_contact_pos = weighted_contact_pos * mask.unsqueeze(-1)  # Mask invalid positions -> (n_contacts, N, n_link_a, n_link_b, 3)

    # Sum weighted positions and contact forces across all contacts (dim=0)
    weighted_sum = torch.sum(weighted_contact_pos, dim=0)  # (N, n_link_a, n_link_b, 3)
    force_sum = torch.sum(contact_force_norm * mask.unsqueeze(-1), dim=0)  # (N, n_link_a, n_link_b, 1)

    # Compute the weighted average (avoid division by zero)
    contact_pos_link = torch.where(
        force_sum > 0,
        weighted_sum / force_sum,
        torch.zeros_like(weighted_sum)
    )
    # Compute validity (at least one valid contact)
    contact_pos_link_valid = (force_sum > 0).squeeze(-1) # (N, n_link_a, n_link_b)

    return contact_pos_link, contact_pos_link_valid


def get_per_geom_a_contact_pos(n_contacts, contact_pos, contact_force_norm, geom_a, geom_b, geom_a_idxs, geom_b_idxs):
    """
    Get all the contact positions between each a_geom all the geoms on b
    Input:
    - contact_pos: (n_contacts, N, 3) tensor of contact positions
    - contact_force_norm: (n_contacts, N) tensor of contact force norms
    - geom_a: (n_contacts, N) tensor of geom indices for entity A
    - geom_b: (n_contacts, N) tensor of geom indices for entity B
    - geom_a_idxs: 1D list of geom indices to filter for entity A
    - geom_b_idxs: 1D list of geom indices to filter for entity B
    Return:
    - contact_pos_geom: (N, n_geom_a, n_contacts, 3) tensor of contact positions
    - contact_pos_geom_valid: (N, n_geom_a, n_contacts) tensor of valid contact positions
    """
    device = contact_pos.device
    mask = get_expanded_mask(n_contacts, geom_a, geom_b, geom_a_idxs, geom_b_idxs)

    assert len(contact_pos.shape) == 3, "contact_pos should have shape (n_contacts, N, 3)"
    n_envs = contact_pos.shape[1]
    n_contacts = contact_pos.shape[0]
    # because there's at most n_contacts across all geoms in each env, we allocate the max size for each geom_a and fill in geom_b contacts
    n_geom_a = len(geom_a_idxs)

    contact_pos_geom = torch.zeros(n_contacts, n_envs, n_geom_a, 3).to(device)  # (n_contacts, N, n_geom_a, 3)
    contact_pos_geom_valid = torch.zeros(n_contacts, n_envs, n_geom_a).to(device)  # (n_contacts, N, n_geom_a)

    # Loop over each geometry_a index to filter and aggregate the contact positions
    for idx in range(n_geom_a):
        collide_b_mask = mask[:, :, idx, :]  # shape (n_contacts, N, n_b)
        any_b_mask = collide_b_mask.any(dim=-1)  # shape (n_contacts, N)
        contact_pos_geom[:, :, idx] = contact_pos * any_b_mask.unsqueeze(-1)  # shape (n_contacts, N, 3)
        contact_pos_geom_valid[:, :, idx] = any_b_mask  # shape (n_contacts, N)

    return contact_pos_geom, contact_pos_geom_valid
