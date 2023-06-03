import rpy2.robjects as robjects
import numpy as np

def read_Sacramento():

    def as_dict(vector):
        """Convert an RPy2 ListVector to a Python dict"""
        result = {}
        for i, name in enumerate(vector.names):
            if isinstance(vector[i], robjects.ListVector):
                result[name] = as_dict(vector[i])
            elif len(vector[i]) == 1:
                result[name] = vector[i][0]
            else:
                result[name] = np.asarray(vector[i])
        return result


    robjects.r['load'](r"C:\Users\lwx\source\repos\RiverDishcharge\RiverDishcharge\Data\Sacramento.rda")
    result = as_dict(robjects.r['Sacramento'])

    return result 


def main():
    read_Sacramento()
     
if __name__ == '__main__':
    main()