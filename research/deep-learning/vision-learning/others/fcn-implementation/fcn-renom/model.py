# Fully Convolutional Networks 8s
class FCN_8s(rm.Model):
    def __init__(self, nb_classes):
        """Class for FCN8. argumnents,
        nb_classeses: number of classes"""
        print("FCN_8s initialized!")
        
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.max_pool1 = rm.MaxPool2d(filter=2, stride=2)
        
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.max_pool2 = rm.MaxPool2d(filter=2, stride=2)
        
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_3 = rm.Conv2d(256, padding=1, filter=3)
        self.max_pool3 = rm.MaxPool2d(filter=2, stride=2)
        
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_3 = rm.Conv2d(512, padding=1, filter=3)
        self.max_pool4 = rm.MaxPool2d(filter=2, stride=2)
        
        self.conv5_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_3 = rm.Conv2d(512, padding=1, filter=3)
        self.max_pool5 = rm.MaxPool2d(filter=2, stride=2)
        
        self.fc6 = rm.Conv2d(4096, padding=3, filter=7)
        self.fc7 = rm.Conv2d(4096, padding=0, filter=1)
        
        self.drop_out = rm.Dropout(0.5)
        
        self.score_fr = rm.Conv2d(nb_classes, filter=1)
        self.upscore2 = rm.Deconv2d(nb_classes, padding=0, filter=2, stride=2)
        self.upscore8 = rm.Deconv2d(nb_classes, padding=0, filter=8, stride=8)
        
        self.score_pool3 = rm.Conv2d(nb_classes, filter=1)
        self.score_pool4 = rm.Conv2d(nb_classes, filter=1)
        
        self.upscore_pool4 = rm.Deconv2d(nb_classes, padding=0, filter=2, stride=2)
        print(help(FCN_8s))

    def forward(self, x):
        t = x
        t = rm.relu(self.conv1_1(t))
        t = rm.relu(self.conv1_2(t))
        self.c1 = t
        t = self.max_pool1(t)
        
        t = rm.relu(self.conv2_1(t))
        t = rm.relu(self.conv2_2(t))
        self.c2 = t
        t = self.max_pool2(t)
        
        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = self.max_pool3(t)
        self.c3 = t
        pool3 = t
        
        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))
        t = self.max_pool4(t)
        self.c4 = t
        pool4 = t
        
        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        self.c5 = t
        t = self.max_pool5(t)
        
        t = rm.relu(self.fc6(t))
        t = self.drop_out(t)
        fc6 = t
        
        t = rm.relu(self.fc7(t))
        fc7 = t
        
        t = self.score_fr(t)
        score_fr = t
        
        t = self.upscore2(t)
        upscore2 = t
        
        t = self.score_pool4(pool4)
        score_pool4 = t
        
        t = upscore2 + score_pool4
        fuse_pool4 = t
        
        t = self.score_pool3(pool3)
        score_pool3 = t
        
        t = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = t
        t = upscore_pool4 + score_pool3
        
        t = self.upscore8(t)
        return t