# Cmpe 493 - Assignment 1

## Name & Surname : Emre Girgin

## ID : 2016400099

## Running : 

```
    python3 edit-distance.py <source_word> <target_word>
```

# Example input : 
```bash
    python3 edit-distance.py "a cat" "an act"
```

# Example output:

``` bash
### Levenshtein Distance ###
Levenshtein Edit Distance from a cat to an act is : 3 

      a  n     a  c  t
  [0, 1, 2, 3, 4, 5, 6]
a [1, 0, 1, 2, 3, 4, 5]
  [2, 1, 1, 1, 2, 3, 4]
c [3, 2, 2, 2, 2, 2, 3]
a [4, 3, 3, 3, 2, 3, 3]
t [5, 4, 4, 4, 3, 3, 3]

Cost: 0 | Copy      | Input: a Output: a
Cost: 1 | Insertion | Input: * Output: n
Cost: 0 | Copy      | Input:   Output:  
Cost: 1 | Replace   | Input: c Output: a
Cost: 1 | Replace   | Input: a Output: c
Cost: 0 | Copy      | Input: t Output: t

------------------------------------
### Damerau Levenshtein Distance ###
Damerau-Levenshtein Edit Distance from a cat to an act is : 2 

      a  n     a  c  t
  [0, 1, 2, 3, 4, 5, 6]
a [1, 0, 1, 2, 3, 4, 5]
  [2, 1, 1, 1, 2, 3, 4]
c [3, 2, 2, 2, 2, 2, 3]
a [4, 3, 3, 3, 2, 2, 3]
t [5, 4, 4, 4, 3, 3, 2]

Cost: 0 | Copy          | Input: a  Output: a
Cost: 1 | Insertion     | Input: * Output: n
Cost: 0 | Copy          | Input:    Output:  
Cost: 1 | Transposition | Input: ca Output: ac
Cost: 0 | Copy          | Input: t  Output: t
```