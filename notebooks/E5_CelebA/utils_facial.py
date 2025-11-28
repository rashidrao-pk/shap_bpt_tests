import torch.nn as nn


# Defining Class for Single Layer. 
class Layer(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super(Layer,self).__init__()
        self.conv=nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding)
        self.bn=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU()
        nn.init.xavier_uniform_(self.conv.weight)
    def forward(self,Input):
        output=self.conv(Input)
        output=self.bn(output)
        output=self.relu(output)
        return output
    


# Complete model
class CelebModel(nn.Module):
    def __init__(self,num_classes=40):
        super(CelebModel,self).__init__()
        
        self.unit1=Layer(in_ch=3,out_ch=32)        
        self.unit2=Layer(in_ch=32,out_ch=32)
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        self.unit3=Layer(in_ch=32,out_ch=64)
        self.unit4=Layer(in_ch=64,out_ch=64)
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.unit5=Layer(in_ch=64,out_ch=128)
        self.unit6=Layer(in_ch=128,out_ch=128)
        self.unit7=Layer(in_ch=128,out_ch=128)
        self.pool3=nn.MaxPool2d(kernel_size=2)
        
        self.unit8=Layer(in_ch=128,out_ch=256,kernel_size=5,padding=0)
        self.unit9=Layer(in_ch=256,out_ch=256,kernel_size=5,padding=0)
        self.unit10=Layer(in_ch=256,out_ch=256,kernel_size=5,padding=0)
        self.pool4=nn.MaxPool2d(kernel_size=2)
        
        self.drop2=nn.Dropout(0.5)   
        
        self.unit11=Layer(in_ch=256,out_ch=512,kernel_size=3,padding=0)
        self.unit12=Layer(in_ch=512,out_ch=512,kernel_size=3,padding=0)
        self.unit13=Layer(in_ch=512,out_ch=512,kernel_size=3,padding=0)
        
        self.pool5=nn.AvgPool2d(kernel_size=2)
        
        self.drop3=nn.Dropout(0.5)
        
        self.model=nn.Sequential(self.unit1,self.unit2,self.pool1,self.unit3,
                                 self.unit4,self.pool2,self.unit5,self.unit6,
                                 self.unit7,self.pool3,self.unit8,self.unit9,
                                 self.unit10,self.pool4,self.drop2,self.unit11,
                                 self.unit12,self.unit13,self.pool5,self.drop3)
        
        self.fc=nn.Linear(in_features=512,out_features=num_classes)
        
    def forward(self,Input):
        
        output=self.model(Input)
        output=output.view(-1,512)
        output=self.fc(output)
        
        return output
    

###################
ground_truth,ground_truths = None,None
union_ground_truths = []
ground_truths = []
brown_hair,black_hair,blond_hair = None,None,None
Eyeglasses = None
def load_ground_truth(att_csv_celeba,intrested_class_lss,img_id=None, verbose=False):
    global ground_truth,ground_truths,union_ground_truth,union_ground_truths
    global brown_hair,Eyeglasses,anno_path

    if verbose:
        print('LAODING GT for ', intrested_class_lss)
    if att_csv_celeba:
        image_id = f'{img_id+1:06}.jpg'
    else:
        image_id = f'{img_id}.jpg'
    union_ground_truths = {}
    ground_truths = []
    for mask_1 in intrested_class_lss:
        ground_truths = []
        if mask_1 == 'Smiling':
            sub_masks = ['l_lip', 'mouth', 'u_lip']
        elif mask_1=='Eyeglasses':
            sub_masks = ['eye_g']
            Eyeglasses = df_attr[df_attr.image_id==image_id]['Eyeglasses'].values[0]==1
        #Black_Hair, Blond_Hair,Brown_Hair,Gray_Hair,Straight_Hair,Wavy_Hair
        elif mask_1=='Brown_Hair':
            brown_hair = df_attr[df_attr.image_id==image_id]['Brown_Hair'].values[0]==1
            sub_masks = ['hair']
        elif mask_1=='Black_Hair':
            black_hair = df_attr[df_attr.image_id==image_id]['Black_Hair'].values[0]==1
            sub_masks = ['hair']
        elif mask_1=='Blond_Hair':
            blond_hair = df_attr[df_attr.image_id==image_id]['Blond_Hair'].values[0]==1
            sub_masks = ['hair']
        elif mask_1=='Gray_Hair':
            brown_hair = df_attr[df_attr.image_id==image_id]['Gray_Hair'].values[0]==1
            sub_masks = ['hair']
        elif mask_1=='Straight_Hair':
            brown_hair = df_attr[df_attr.image_id==image_id]['Straight_Hair'].values[0]==1
            sub_masks = ['hair']
        elif mask_1=='Wavy_Hair':
            brown_hair = df_attr[df_attr.image_id==image_id]['Wavy_Hair'].values[0]==1
            sub_masks = ['hair']
               
        else:
            sub_masks = [mask_1]
        
        # Loop over each sub-mask
        for sub_mas in sub_masks:
            
            folder = find_folder(img_id, anno_pth)
            anno_subpath = os.path.join(ds_anno_path,str(folder))
            anno_filename = f'{os.path.join(anno_subpath, f"{img_id:05}_{sub_mas}")}.png'
            #print('sub_masks:',sub_masks,anno_filename,os.path.exists(anno_filename))
            # print(anno_filename,img_id,image_id,os.path.exists(anno_filename), brown_hair, Eyeglasses, df_attr[df_attr.image_id==image_id]['Eyeglasses'])
            #if not os.path.exists(anno_filename) or not brown_hair or not Eyeglasses and load_positives_only:
            if not os.path.exists(anno_filename):
                ground_truth = np.zeros((224,224)).astype(int)
            else:
                #print('anno_subpath: \t',anno_filename,img_id,mask_1)
                ground_truth = cv2.imread(anno_filename, cv2.IMREAD_COLOR)[:, :, ::-1]
                ground_truth = cv2.resize(ground_truth, [224, 224], interpolation=cv2.INTER_NEAREST)
                ground_truth = ground_truth[:,:,0].astype(int)
            ground_truths.append(ground_truth)
        
        # Calculate the union of the ground truths for the current mask
        if len(ground_truths) > 0:
            union_ground_truth = ground_truths[0]
            for ground_truth in ground_truths[1:]:
                union_ground_truth = cv2.bitwise_or(union_ground_truth, ground_truth)
        else:
            union_ground_truth = ground_truths[0]
        
        # Append the union ground truth to the list of all unions
        # union_ground_truths.append(union_ground_truth)
        union_ground_truths[mask_1] = union_ground_truth



        # # evaluate the ground truth mask with the background replacement strategy for masking function
        # predicted_fG = f_masked(np.expand_dims(ground_truth, axis=0))[0]
        # f_G = float(predicted_fG[predicted_cls])
        # print(class_names[predicted_cls], f_G, predicted_cls, f_G)
        # print('softmax prob:', np_softmax(predicted_fG)[predicted_cls])

        # # evaluate the backgrounf (negative of the ground truth mask)
        # background_mask = np.logical_not(ground_truth)
        # predicted_fB = f_masked(np.expand_dims(background_mask, axis=0))[0]
        # f_B = float(predicted_fB[predicted_cls])
        # print(class_names[predicted_cls], f_B, predicted_cls, f_B)
        # print('softmax prob:', np_softmax(predicted_fB)[predicted_cls])

        # print()
        # print('nu(S):  ', round(f_S, 4))
        # print('nu(G):  ', round(f_G, 4))
        # print('nu(S/G):', round(f_B, 4))
        # print('nu(0):  ', round(f_0, 4))




def natural_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]