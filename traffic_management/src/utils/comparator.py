

def movement_comparator(m1, m2):
    direction_sort_order = {'T': 0, 'L': 1, 'S': 2, 'R': 3}

    m1_0 = int(m1[0])
    m2_0 = int(m2[0])

    if m1_0 > m2_0:
        return 1
    if m1_0 < m2_0:
        return -1

    m1_1 = m1[1]
    m2_1 = m2[1]

    if direction_sort_order[m1_1] > direction_sort_order[m2_1]:
        return 1
    if direction_sort_order[m1_1] < direction_sort_order[m2_1]:
        return -1

    m1_len = len(m1)
    m2_len = len(m2)

    if m1_len < m2_len:
        if m1_1 == "R":
            return 1
        else:
            return -1

    if m1_len > m2_len:
        if m1_1 == "R":
            return -1
        else:
            return 1

    if m1_len > 2:

        m1_end = int(m1[2:])
        m2_end = int(m2[2:])

        if m1_end > m2_end:
            return 1
        if m1_end < m2_end:
            return -1

    return 0


def phase_comparator(p1, p2):

    p1_split = p1.split('_')
    p2_split = p2.split('_')

    p1_split_len = len(p1_split)
    p2_split_len = len(p2_split)

    if p1_split_len < p2_split_len:
        return 1
    if p1_split_len > p2_split_len:
        return -1

    for i in range(0, p1_split_len):
        p1_mi = p1_split[i]
        p2_mi = p2_split[i]

        movement_comparison = movement_comparator(p1_mi, p2_mi)

        if movement_comparison != 0:
            return movement_comparison

    return 0


def location_comparator(p1, p2):

    p1_x, p1_y = p1.coords[0]
    p2_x, p2_y = p2.coords[0]

    if p1_y < p2_y:
        return 1
    if p1_y > p2_y:
        return -1

    if p1_x < p2_x:
        return -1
    if p1_x > p2_x:
        return 1

    return 0


def direction_precedence_comparator(d1, d2):

    direction_sort_order = {'t': 2, 'l': 1, 's': 0, 'r': 1}

    if direction_sort_order[d1] > direction_sort_order[d2]:
        return 1
    if direction_sort_order[d1] < direction_sort_order[d2]:
        return -1

    return 0


def phases_subset_comparator(p1, p2):

    p1_set = set(p1.split('_'))
    p2_set = set(p2.split('_'))

    if p1_set.issubset(p2_set):
        if p2_set.issubset(p1_set):
            return 0
        else:
            return 1
    else:
        if p2_set.issubset(p1_set):
            return -1
        else:
            return
