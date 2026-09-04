"""Small helpers whose semantics have to match the Rust implementation in core."""


def last_argmax(values: list[float]) -> int:
    """Index of the maximum value, resolving ties in favour of the *last* maximum.

    Mirrors Rust's `Iterator::max_by_key`, which core uses to pick MMR candidates.
    `np.argmax` returns the first maximum instead, which orders exact ties differently.
    """
    best_index = 0
    for index in range(1, len(values)):
        if values[index] >= values[best_index]:
            best_index = index
    return best_index


def swap_remove(items: list[int], position: int) -> int:
    """Remove and return `items[position]`, moving the last item into the freed slot.

    Mirrors `IndexSet::swap_remove`, which core uses to drop a selected MMR candidate,
    and which therefore decides the order the remaining candidates are visited in.
    """
    value = items[position]
    items[position] = items[-1]
    items.pop()
    return value
