import time

class Range():
    def __init__(self, vals=[], incs=None):
        # A dictionary of includes for the specific points
        self.includes = {}
        # A list of ranges, alternating between included and not included
        self.values = []
        self.add_range(vals, incs)

    # * Main functions to simplify editing
    def remove_range(self, vals: iter, incs=None):
        self._edit_range(vals, incs, is_removal=True)
    def add_range(self, vals: iter, incs=None):
        self._edit_range(vals, incs, is_removal=False)

    # * Function that will add or remove a range from the space
    def _edit_range(self, vals: iter, incs=None, is_removal=False):
        # Default value for includes
        if not incs:
            incs = {val: True for val in vals}
        else:
            if isinstance(incs, list):
                incs = {val: inc for val, inc in zip(vals, incs)}
            # Ensure it is given properly
            assert len(incs) == len(vals), f'Include parameters must be the same length: incs-{len(incs)}, vals-{len(vals)}. Values might include duplicate entries'
        assert len(vals) % 2 == 0, 'Vals must be in groups of 2'

        # Iterate 2 at a time
        for b, e in self._group_list(vals):
            # Extends beginning of range
            b_in_range, b_index = self._contains(b)
            e_in_range, e_index = self._contains(e)
            # Delete points in between our range since they are now meaningless
            if b_index != e_index:
                self._delete_points(b_index, e_index)
            # If b/e was in the range that means it was useless (or useful depending on whether we are removing)
            if not (b_in_range ^ is_removal):
                self.values.append(b)
                self.includes[b] = incs[b]
            # Do the same for e
            if not (e_in_range ^ is_removal):
                self.values.append(e)
                self.includes[e] = incs[e]
            # Sort the data
            self.values = sorted(self.values)
    
    # Deletes a range between to values
    def _delete_points(self, val1: float, val2: float):
        for i in range(val1, val2):
            val = self.values[i]
            print('DELETING', val)
            del self.includes[val]
            del self.values[i]

    # Returns the list, but n-size groups
    def _group_list(self, vals, n=2):
        return zip(*[iter(vals)]*n)

    # Returns the condition of whether the point is in the range and what index it would be at
    def _contains(self, point: float) -> tuple:
        # Create a new array with the point and find its index
        self.values.append(point)
        self.values = sorted(self.values)
        # If it's index is odd, it's splitting up a range meaning its included
        i = self.values.index(point)
        # Remove the item
        self.values.remove(point)
        # Cover possibility of being on the edge
        if point in self.values:
            return self.includes[point], i
        return (bool(i % 2), i)

    def __contains__(self, point: float) -> bool:
        return bool(self._contains(point)[0])

    # * PRINT OUT
    def __str__(self) -> str:
        print_out = ''

        for i, val in enumerate(self.values):
            if i % 2 == 0:
                if self.includes[val]:
                    print_out += '['
                else:
                    print_out += '('    
                print_out += f'{val}, '
            
            else:
                print_out += f'{val}'

                if self.includes[val]:
                    print_out += ']'
                else:
                    print_out += ')'
                
                if self.values[i] != self.values[-1]:
                    print_out += ' U '

        return print_out


if __name__ == "__main__":
    r = Range([0, 10])
    # r.remove_range([1, 5])
    print(r)