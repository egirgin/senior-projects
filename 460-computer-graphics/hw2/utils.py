width = 1000
height = 1000

scale = 50

def raster2screen(row, column):
    # Raster space to NDC space
    new_row = (row + 0.5) / width
    new_column = (column + 0.5) / height

    # NDC space to screen space
    new_row = 2 * new_row -1
    new_column = 1 - 2 * new_column

    # Scale
    new_row *= scale
    new_column *= scale

    return new_row, new_column