def string_reverse1(string):
    return string[::-1]

if __name__ == '__main__':
    import cPickle
    train_data_x_zheng = []

    train_data_y = []
    test_data_x_zheng = []

    test_data_y = []

    train_vec_file1 = open("yuliao/Fbasic/train_seq.lstm", 'r')
    train_label_file = open("yuliao/Fbasic/Ftrainlabel.txt", 'r')
    test_vec_file1 = open("yuliao/Fbasic/test_seq.lstm", 'r')
    test_label_file = open("yuliao/Fbasic/Ftestlabel.txt", 'r')

    for i in range(0, 47818):
        vec_line1 = train_vec_file1.readline()
        vec_line1 = str.strip(vec_line1)
        vec_line1 = vec_line1[0:len(vec_line1)]
        label_line = train_label_file.readline()
        train_data_x_zheng.append([int(elem) for elem in vec_line1.split(' ')])

        if label_line[0] == '+':
            train_data_y.extend([1])

        else:
            train_data_y.extend([0])



    for i in range(0, 11954):
        vec_line1 = test_vec_file1.readline()
        vec_line1 = str.strip(vec_line1)
        vec_line1 = vec_line1[0:len(vec_line1)]
        label_line = test_label_file.readline()
        test_data_x_zheng.append([int(elem) for elem in vec_line1.split(' ')])
        # train_data_x_zheng.append([int(elem) for elem in vec_line1.split(' ')])
        if label_line[0] == '+':
            test_data_y.extend([1])

        else:
            test_data_y.extend([0])


    output_file = open("Fscope_basic.pkl", 'w')

    train_data = [train_data_x_zheng, train_data_y]
    test_data = [test_data_x_zheng, test_data_y]
    cPickle.dump(train_data, output_file)
    cPickle.dump(test_data, output_file)

    output_file.close()
