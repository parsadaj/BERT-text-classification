from utils import *
import sys


def main():
    try:
        data_path, out_dir = sys.argv[1:]
    except ValueError:
        HW_path = os.getcwd()
        data_path, out_dir = os.path.join(HW_path, 'data', 'persica.csv'), os.path.join(HW_path, 'results')
    
    create_datasets(data_path, out_dir)


if __name__ == '__main__':
    main()