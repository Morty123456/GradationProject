def classification(predict, test):
    total = len(predict)
    count = 0
    res_predict = []
    for i in range(len(predict)):
        predict_local = 0
        test_local = 0
        for j in range(len(predict[i])):
            if predict[i][j] > predict[i][predict_local]:
                predict_local = j
            if test[i][j] > test[i][test_local]:
                test_local = j
            res_predict.append(predict_local)
        if predict_local == test_local:
            count += 1
    return count/total, res_predict

def classification_detailded(predict, test):
    predict_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    test_count = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    correctClassification = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    total = len(predict)
    count = 0
    for i in range(len(predict)):
        predict_local = 0
        test_local = 0
        for j in range(len(predict[i])):
            if predict[i][j] > predict[i][predict_local]:
                predict_local = j
            if test[i][j] > test[i][test_local]:
                test_local = j
        predict_count[predict_local] += 1
        test_count[test_local] += 1
        if predict_local == test_local:
            correctClassification[predict_local] += 1
            count += 1
    score = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(predict_count)):
        score[i] = predict_count[i]/test_count[i]
    # 分类准确率、每种类别的准确率、每种类别的识别数、每种类别的实际数、每种类别被正确识别的数目
    return count/total, score, predict_count, test_count, correctClassification