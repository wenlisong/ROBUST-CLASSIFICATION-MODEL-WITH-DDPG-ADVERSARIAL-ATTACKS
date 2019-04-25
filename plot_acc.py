import sys
from matplotlib import pyplot as plt

def read_acc_from_txt(file=None):
    res = []
    with open(file, 'r') as f:
        for i in range(110):
            val = float(f.readline().strip().split('=')[-1].strip()) * 100
            res.append(val)
    return res

def main(argvs):
    data1 = read_acc_from_txt('./result/original_train_accuracy.txt')
    data2 = read_acc_from_txt('./result/original_noise_train_accuracy.txt')
    data3 = read_acc_from_txt('./result/robust_train_accuracy.txt')
    data4 = read_acc_from_txt('./result/robust_noise_train_accuracy.txt')

    plt.title('accracy')
    plt.xlabel('class')
    plt.ylabel('accuracy(%)')
    plt.plot(range(0,110,1), data1,'cyan', label='GoogLeNet')
    plt.plot(range(0, 110, 1), data2, 'b', label='GoogLeNet under Noise(stddev=0.2)')
    plt.plot(range(0, 110, 1), data3, 'orange', label='Robust GoogLeNet')
    plt.plot(range(0, 110, 1), data4, 'r', label='Robust GoogLeNet under Noise(stddev=0.2)')
    
    plt.legend(bbox_to_anchor=[0.6, 0.95])
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    main(sys.argv[1:])