from random import sample


def order(x, NoneIsLast=True, decreasing=False):
    """
    Returns the ordering of the elements of x. The list
    [ x[j] for j in order(x) ] is a sorted version of x.

    Missing values in x are indicated by None. If NoneIsLast is true,
    then missing values are ordered to be at the end.
    Otherwise, they are ordered at the beginning.
    """
    omitNone = False
    if NoneIsLast == None:
        NoneIsLast = True
        omitNone = True

    n = len(x)
    ix = list(range(n))
    if None not in x:
        ix.sort(reverse=decreasing, key=lambda j: x[j])
    else:
        # Handle None values properly.
        def key(i, x=x):
            elem = x[i]
            # Valid values are True or False only.
            if decreasing == NoneIsLast:
                return not (elem is None), elem
            else:
                return elem is None, elem

        ix = range(n)
        ix.sort(key=key, reverse=decreasing)

    if omitNone:
        n = len(x)
        for i in range(n - 1, -1, -1):
            if x[ix[i]] == None:
                n -= 1
        return ix[:n]
    return ix


def rank(x, NoneIsLast=True, decreasing=False, ties="first"):
    """
    Returns the ranking of the elements of x. The position of the first
    element in the original vector is rank[0] in the sorted vector.

    Missing values are indicated by None.  Calls the order() function.
    Ties are NOT averaged by default. Choices are:
         "first" "average" "min" "max" "random" "average"
    """
    omitNone = False
    if NoneIsLast == None:
        NoneIsLast = True
        omitNone = True
    O = order(x, NoneIsLast=NoneIsLast, decreasing=decreasing)
    R = O[:]
    n = len(O)
    for i in range(n):
        R[O[i]] = i
    if ties == "first" or ties not in ["first", "average", "min", "max", "random"]:
        return R

    blocks = []
    isnewblock = True
    newblock = []
    for i in range(1, n):
        if x[O[i]] == x[O[i - 1]]:
            if i - 1 not in newblock:
                newblock.append(i - 1)
            newblock.append(i)
        else:
            if len(newblock) > 0:
                blocks.append(newblock)
                newblock = []
    if len(newblock) > 0:
        blocks.append(newblock)

    for i, block in enumerate(blocks):
        # Don't process blocks of None values.
        if x[O[block[0]]] == None:
            continue
        if ties == "average":
            s = 0.0
            for j in block:
                s += j
            s /= float(len(block))
            for j in block:
                R[O[j]] = s
        elif ties == "min":
            s = min(block)
            for j in block:
                R[O[j]] = s
        elif ties == "max":
            s = max(block)
            for j in block:
                R[O[j]] = s
        elif ties == "random":
            s = sample([O[i] for i in block], len(block))
            for i, j in enumerate(block):
                R[O[j]] = s[i]
        else:
            for i, j in enumerate(block):
                R[O[j]] = j
    if omitNone:
        R = [R[j] for j in range(n) if x[j] != None]
    return R



