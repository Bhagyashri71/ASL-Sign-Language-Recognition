###### Healper functions like plotting, confusion matrix ########


def label_to_char(index):
    return chr(ord('A') + index)

def char_to_label(char):
    return ord(char.upper()) - 65
