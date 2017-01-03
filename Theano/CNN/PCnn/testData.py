class InstanceBag(object):

    def __init__(self, entities, rel, num, sentences, positions, entitiesPos):
        self.entities = entities
        self.rel = rel
        self.num = num
        self.sentences = sentences
        self.positions = positions
        self.entitiesPos = entitiesPos
import theano.sandbox.cuda
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))
def readData(filename):
    f = open(filename, 'r')
    data = []
    while 1:
        line = f.readline()
        if not line:
            break
        entities = map(int, line.split(' '))
        line = f.readline()
        bagLabel = line.split(' ')

        rel = map(int, bagLabel[0:-1])
        num = int(bagLabel[-1])
        positions = []
        sentences = []
        entitiesPos = []
        for i in range(0, num):
            sent = f.readline().split(' ')
            positions.append(map(int, sent[0:2]))
            epos = map(int, sent[0:2])
            epos.sort()
            entitiesPos.append(epos)
            sentences.append(map(int, sent[2:-1]))
        ins = InstanceBag(entities, rel, num, sentences, positions, entitiesPos)
        data += [ins]
    f.close()
    return data
def select_instance(rels, nums, sents, poss, eposs, img_h):
    import numpy as np
    numBags = len(rels)
    print numBags
    print(nums)
    x = np.zeros((numBags, img_h), dtype='int32')
    p1 = np.zeros((numBags, img_h), dtype='int32')
    p2 = np.zeros((numBags, img_h), dtype='int32')
    pool_size = np.zeros((numBags, 2), dtype='int32')
    y = np.asarray(rels, dtype='int32')

    for bagIndex, insNum in enumerate(nums):
        maxIns = 0
        maxP = -1
        if insNum > 1:
            for m in range(insNum):
                insPos = poss[bagIndex][m]
                insX = np.asarray(sents[bagIndex][m], dtype='int32').reshape((1, img_h))
                insPf1 = np.asarray(insPos[0], dtype='int32').reshape((1, img_h))
                insPf2 = np.asarray(insPos[1], dtype='int32').reshape((1, img_h))
                insPool = np.asarray(eposs[bagIndex][m], dtype='int32').reshape((1, 2))
                print(insPos)
                # results = test_one(insX, insPf1, insPf2, insPool)
                # tmpMax = results.max()
                # if tmpMax > maxP:
                #     maxIns = m
                #     maxP=tmpMax
            # else:
            #     maxIns = 0
        x[bagIndex,:] = sents[bagIndex][maxIns]
        p1[bagIndex,:] = poss[bagIndex][maxIns][0]
        p2[bagIndex,:] = poss[bagIndex][maxIns][1]
        pool_size[bagIndex,:] = eposs[bagIndex][maxIns]
    return [x, p1, p2, pool_size, y]
def bags_decompose(data_bags):
    bag_sent = [data_bag.sentences for data_bag in data_bags]
    bag_pos = [data_bag.positions for data_bag in data_bags]
    bag_num = [data_bag.num for data_bag in data_bags]
    bag_rel = [data_bag.rel for data_bag in data_bags]
    bag_epos = [data_bag.entitiesPos for data_bag in data_bags]
    return [bag_rel, bag_num, bag_sent, bag_pos, bag_epos]
def testData():
    from data2cv import make_idx_data_cv
    data =readData('train_filtered.data')
    newdata =make_idx_data_cv(data,3,15)
    [train_rels, train_nums, train_sents, train_poss, train_eposs] = bags_decompose(newdata)
    pool_size = np.zeros((len(train_rels), 2), dtype='int32')
    for bagIndex, insNum in enumerate(train_nums):
        print('bag '+str(bagIndex))
        print('isNum '+str(insNum))
        pool_size[bagIndex,:] = train_eposs[bagIndex][0]
    print pool_size


    print newdata[0].positions
    print( newdata[0].sentences)
    print(newdata[0].entitiesPos)
    # select_instance(train_rels, train_nums, train_sents, train_poss, train_eposs,9)
if __name__=='__main__':
    import numpy as np
    import theano

    testData()
    def createPFMatrix():
        rng = np.random.RandomState(3435)
        PF1 = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]), dtype=theano.config.floatX)
        padPF1 = np.zeros((1, 5))
        PF1 = np.vstack((padPF1, PF1))
        print(PF1)
        print(PF1.shape)

    # f =open('gap_40_len_80/test_filtered.data','r')
    # data = []
    # while 1:
    #     line = f.readline()
    #     if not line:
    #         break
    #     entities = map(int, line.split(' '))
    #     line = f.readline()
    #     bagLabel = line.split(' ')
    #
    #     rel = map(int, bagLabel[0:-1])
    #     num = int(bagLabel[-1])
    #     positions = []
    #     sentences = []
    #     entitiesPos = []
    #     for i in range(0, num):
    #         sent = f.readline().split(' ')
    #         positions.append(map(int, sent[0:2]))
    #         epos = map(int, sent[0:2])
    #         epos.sort()
    #         entitiesPos.append(epos)
    #         sentences.append(map(int, sent[2:-1]))
    #     ins = InstanceBag(entities, rel, num, sentences, positions, entitiesPos)
    #     data += [ins]
    # f.close()
    # print len(data)
