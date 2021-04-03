"""
__author__: Emre Girgin
__date__  : 17.11.2020
__email__ : emre.girgin@boun.edu.tr
"""


import argparse

operationsStack = []

parser = argparse.ArgumentParser()

parser.add_argument("source_word")
parser.add_argument("target_word")

args = parser.parse_args()


s1 =  str(args.source_word)  # rows
s2 = str(args.target_word) # columns

###################################### LEVENSHTEIN DISTANCE ###############################################################

# Recursion version (To check if dynamic programming method is works properly)
def calculate_distance(string1, string2):
    if len(string1) == 0:
        return len(string2)
    elif len(string2) == 0:
        return len(string1)
    else:
        return min(
            calculate_distance(string1[:-1], string2) + 1,
            calculate_distance(string1, string2[:-1]) + 1,
            calculate_distance(string1[:-1], string2[:-1]) + (0 if string1[-1] == string2[-1] else 1)
        )

# Print Edit Table similar to the one in the lecture slides
def print_edit_table(matrix, s1, s2):

    # Not used
    def extend(letter):
        return letter + "  "
    
    # Not Used
    def listToString(myList):
        string = ""

        for item in myList:
            string += item

        return string
    
    #print("     ", listToString(list(map(extend, s2))))
    print("     ", "  ".join(s2))
    for i, row in enumerate(matrix):
        if i == 0 :
            print(" ", row)
        else: 
            print(s1[i-1], row)


# Dynamic Programming
def levenshteinDistance(s1, s2):
    distanceMatrix = []

    for i in range(len(s1)+1):
        distanceMatrix.append([ 0 for _ in range(len(s2)+1) ])
        distanceMatrix[i][0] = i
    for i in range(len(s2)+1):
        distanceMatrix[0][i] = i
    for i in range(1,len(s1)+1):
        for j in range(1,len(s2)+1):
            if s1[i-1] == s2[j-1]:
                distanceMatrix[i][j] = min(
                    distanceMatrix[i-1][j] + 1, # Delete
                    distanceMatrix[i][j-1] + 1, # Insert
                    distanceMatrix[i-1][j-1] # Copy
                )
            else:
                distanceMatrix[i][j] = min(
                    distanceMatrix[i-1][j] + 1, # Delete
                    distanceMatrix[i][j-1] + 1, # Insert
                    distanceMatrix[i-1][j-1] + 1 # Replace
                )
    return distanceMatrix


# Sequence of Operations needed to Transform source to target word
def findOperations(matrix, row, column):

    leftNeighbor = matrix[row][column-1]
    cornerNeighbor = matrix[row-1][column-1]
    upperNeighbor = matrix[row-1][column]

    # Try to reach 0,0 point
    if row == 0 and column == 0:
        return

    # First check if copy is possible
    if min(leftNeighbor, cornerNeighbor, upperNeighbor) == cornerNeighbor:
        if matrix[row][column] == cornerNeighbor:
            operationsStack.append("0/Copy     /{}/{}".format(str(row), str(column)))
        else:
            operationsStack.append("1/Replace  /{}/{}".format(str(row), str(column)))
        findOperations(matrix, row-1, column-1)
    # Insertion and Deletion have the same cost
    elif min(leftNeighbor, cornerNeighbor, upperNeighbor) == leftNeighbor:
        operationsStack.append("1/Insertion/{}/{}".format("*", str(column)))
        findOperations(matrix, row, column-1)
    elif min(leftNeighbor, cornerNeighbor, upperNeighbor) == upperNeighbor:
        operationsStack.append("1/Deletion /{}/{}".format(str(row), "*"))
        findOperations(matrix, row-1, column)

# Print operations needed
def parseStack():
    operationsStack.reverse()

    for operation in operationsStack:
        items = operation.split("/")
        cost = items[0]
        op = items[1]
        inp = items[2]
        if not inp == "*":
            inp = s1[int(inp)-1]
        out = items[3]
        if not out == "*":
            out = s2[int(out)-1]

        print("Cost: " + cost + " | " + op + " | Input: " + inp + " Output: " + out)


print("### Levenshtein Distance ###")

levenshteinDistanceMatrix = levenshteinDistance(s1, s2)

levenshteinDistanceNumber = levenshteinDistanceMatrix[-1][-1]
print("Levenshtein Edit Distance from {} to {} is : {} \n".format(s1, s2, levenshteinDistanceNumber) )

print_edit_table(levenshteinDistance(s1, s2), s1, s2)

findOperations(levenshteinDistanceMatrix, int(len(s1)), int(len(s2)))

print()

parseStack()

print()

###################################### DAMERAU LEVENSHTEIN DISTANCE ###############################################################

operationsStack = []

# Recursion (To check if dynamic programming method is works properly)
def calculate_distance(string1, string2):
    if len(string1) == 0:
        return len(string2)
    elif len(string2) == 0:
        return len(string1)
    else:
        return min(
            min(
                calculate_distance(string1[:-1], string2) + 1,
                calculate_distance(string1, string2[:-1]) + 1,
                calculate_distance(string1[:-1], string2[:-1]) + (0 if string1[-1] == string2[-1] else 1)
            ), 
            calculate_distance(string1[:-2], string2[:-2]) + (1 if ((len(string1) > 1 and len(string2) > 1) and (string1[-1] == string2[-2] and string1[-2] == string2[-1])) else 9999999999999))



# Dynamic Programming
def damerauLevenshteinDistance(s1, s2):
    distanceMatrix = []

    for i in range(len(s1)+1):
        distanceMatrix.append([ 0 for _ in range(len(s2)+1) ])
        distanceMatrix[i][0] = i
    for i in range(len(s2)+1):
        distanceMatrix[0][i] = i
    for i in range(1,len(s1)+1):
        for j in range(1,len(s2)+1):
            if s1[i-1] == s2[j-1]:
                distanceMatrix[i][j] = min(
                    distanceMatrix[i-1][j] + 1, # Delete
                    distanceMatrix[i][j-1] + 1, # Insert
                    distanceMatrix[i-1][j-1] # Copy
                )
            else:
                distanceMatrix[i][j] = min(
                    distanceMatrix[i-1][j] + 1, # Delete
                    distanceMatrix[i][j-1] + 1, # Insert
                    distanceMatrix[i-1][j-1] + 1 # Replace
                )
            try:
                # check if transposition is possible
                if ( (i > 1) and (j > 1) ) and ( (s1[i-1] == s2[j-2]) and (s1[i-2] == s2[j-1]) ):
                    distanceMatrix[i][j] = min(
                        distanceMatrix[i][j],
                        distanceMatrix[i-2][j-2] + 1
                    )
            except Exception as e:
                print(e)
                print("error",i,j)

    return distanceMatrix

# Print Edit Table similar to the one in the lecture slides
def print_edit_table(matrix, s1, s2):

    # To print table visually better (Not used)
    def extend(letter):
        return letter + "  "
    
    # Constructs a string from a list (Not used)
    def listToString(myList):
        string = ""

        for item in myList:
            string += item

        return string
    
    #print("     ", listToString(list(map(extend, s2))))
    print("     ", "  ".join(s2))
    for i, row in enumerate(matrix):
        if i == 0 :
            print(" ", row)
        else: 
            print(s1[i-1], row)



# Sequence of Operations needed to Transform source to target word
def findOperations(matrix, row, column):

    leftNeighbor = matrix[row][column-1]
    cornerNeighbor = matrix[row-1][column-1]
    upperNeighbor = matrix[row-1][column]

    # Try to reach 0,0 point
    if row == 0 and column == 0:
        return

    if (row > 1 and column > 1) and (s1[row-1] == s2[column-2] and s1[row-2] == s2[column-1]) and (not s1[row-1] == s1[row-2]):

        transpositionNeighbor = matrix[row-2][column-2]
        
        # Try transposition second
        if min(leftNeighbor, cornerNeighbor, upperNeighbor, transpositionNeighbor) == transpositionNeighbor:
            operationsStack.append("1/Transposition/{}/{}".format(str(row), str(column)))
            findOperations(matrix, row-2, column-2)


    else:
        
        # First check if copy is possible
        if min(leftNeighbor, cornerNeighbor, upperNeighbor) == cornerNeighbor:
            if matrix[row][column] == cornerNeighbor:
                operationsStack.append("0/Copy         /{}/{}".format(str(row), str(column)))
            else:
                operationsStack.append("1/Replace      /{}/{}".format(str(row), str(column)))
            findOperations(matrix, row-1, column-1)
        
        # Insertion and Deletion have the same cost
        elif min(leftNeighbor, cornerNeighbor, upperNeighbor) == leftNeighbor:
            operationsStack.append("1/Insertion    /{}/{}".format("*", str(column)))
            findOperations(matrix, row, column-1)
        elif min(leftNeighbor, cornerNeighbor, upperNeighbor) == upperNeighbor:
            operationsStack.append("1/Deletion     /{}/{}".format(str(row), "*"))
            findOperations(matrix, row-1, column)

# Print operations needed
def parseStack():
    operationsStack.reverse()

    for operation in operationsStack:
        items = operation.split("/")
        cost = items[0]
        op = items[1]
        inp = items[2]

        # If not insertion
        if not inp == "*":
            if op == "Transposition":
                inp = s1[int(inp)-2 : int(inp)]
            else:
                inp = s1[int(inp)-1] + " "
        
        out = items[3]
        
        # If not deletion
        if not out == "*":
            if op == "Transposition":
                out = s2[int(out)-2 : int(out)]
            else:
                out = s2[int(out)-1]
            

        print("Cost: " + cost + " | " + op + " | Input: " + inp + " Output: " + out)


print("------------------------------------")
print("### Damerau Levenshtein Distance ###")


damerauLevenshteinDistanceMatrix = damerauLevenshteinDistance(s1, s2)

damerauLevenshteinDistanceNumber = damerauLevenshteinDistanceMatrix[-1][-1]

print("Damerau-Levenshtein Edit Distance from {} to {} is : {} \n".format(s1, s2, damerauLevenshteinDistanceNumber) )

print_edit_table(damerauLevenshteinDistance(s1, s2), s1, s2)

findOperations(damerauLevenshteinDistanceMatrix, int(len(s1)), int(len(s2)))

print()

parseStack()

print()
