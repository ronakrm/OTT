# Utilities etc.

def serialize_to_file(file, losses):
    file=open(file, 'w')
    for l in losses:
        file.write("{0}\n".format(l))
    file.close()