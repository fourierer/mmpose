import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

json_file_path = '/mnt/cfs/algorithm/users/zheng.sun/source_code/mmpose/work_dirs/res50_coco_256x192/test.json'
json_file = open(json_file_path, mode='w')
 
'''
save_json_content = []
for img_name in img_name_list:
    result_json = {
        "image_name": img_name,
        "category": 1,
        "score": 0.99074}
    save_json_content.append(result_json)
'''
# save_json_content = [{'score':(1,2,3)},{'keypoints':34}]
save_json_content = [np.array([2,3]),np.array([4,5]),1,2.3] # 遇到list或者dict中有np.array，int等类型的，需要加一个NpEncoder类才可以输入到json文件中
with open(json_file_path,'w') as fid:
    json.dump(save_json_content, fid, cls=NpEncoder)
# json.dump(save_json_content, json_file, ensure_ascii=False, indent=4) # 保存中文

with open(json_file_path,'r') as fid:
    content = json.load(fid)
    print(content)


outputs = [(np.array([[[5.8808624e+02, 2.9440927e+02, 4.0144362e-02],
        [5.8825696e+02, 2.9332803e+02, 2.6917407e-02],
        [5.8825696e+02, 2.9332803e+02, 1.4662072e-02],
        [5.9189905e+02, 2.9332803e+02, 4.5079391e-02],
        [5.9559808e+02, 3.0180728e+02, 3.4371302e-02],
        [5.9275269e+02, 2.9782373e+02, 1.0120705e-01],
        [5.9445990e+02, 2.9748230e+02, 1.2720147e-01],
        [5.9081781e+02, 3.0123819e+02, 4.4391863e-02],
        [5.9059021e+02, 3.0180728e+02, 4.0340431e-02],
        [5.9593951e+02, 2.9645795e+02, 1.7776933e-02],
        [5.9394775e+02, 3.0744113e+02, 1.3803283e-02],
        [5.9349249e+02, 3.0744113e+02, 3.9882865e-02],
        [5.9349249e+02, 3.0744113e+02, 3.2823235e-02],
        [5.9332178e+02, 2.9839282e+02, 7.7666596e-02],
        [5.9445990e+02, 2.9702704e+02, 6.4865068e-02],
        [5.9457373e+02, 3.0533554e+02, 5.1532548e-02],
        [5.9212671e+02, 3.0744113e+02, 5.1316544e-02]]], dtype=np.float32), 
        np.array([[5.9303723e+02, 3.0038458e+02, 5.4631285e-02, 7.2841711e-02, 1.5917746e+02, 2.2374015e-02]], dtype=np.float32), 
        ['d', 'a', 't', 'a', '/', 'c', 'o', 'c', 'o', '/', 't', 'e', 's', 't', '2', '0', '1', '7', '/', '0', '0', '0', '0', '0', '0', '0', '7', '3', '7', '2', '3', '.', 'j', 'p', 'g'],
        None),
        (np.array([[[5.08394806e+02, 2.92105804e+02, 1.47382349e-01],
        [5.08722015e+02, 2.91206024e+02, 1.55548319e-01],
        [5.08558411e+02, 2.91124207e+02, 1.19948968e-01],
        [5.09539978e+02, 2.90797028e+02, 2.05885082e-01],
        [5.09376404e+02, 2.91042419e+02, 8.12930986e-02],
        [5.10930542e+02, 2.93823547e+02, 6.86325654e-02],
        [5.09294586e+02, 2.93987152e+02, 4.96294014e-02],
        [5.12566528e+02, 2.97422668e+02, 3.68483514e-02],
        [5.10194366e+02, 2.98976807e+02, 4.90388907e-02],
        [5.12239319e+02, 2.98567841e+02, 1.88152380e-02],
        [5.09212799e+02, 2.92351196e+02, 1.29364645e-02],
        [5.12239319e+02, 3.00040192e+02, 1.86116435e-02],
        [5.09539978e+02, 3.00285583e+02, 1.73258763e-02],
        [5.09948975e+02, 2.95868500e+02, 2.08466202e-02],
        [5.09458191e+02, 2.94641541e+02, 2.45963801e-02],
        [5.12321106e+02, 2.99140411e+02, 2.88237128e-02],
        [5.09989868e+02, 3.00490082e+02, 4.39075381e-02]]], dtype=np.float32),
        np.array([[5.1064426e+02, 2.9541861e+02, 3.9262891e-02, 5.2350521e-02, 8.2217308e+01, 1.9711582e-02]], dtype=np.float32), 
        ['d', 'a', 't', 'a', '/', 'c', 'o', 'c', 'o', '/', 't', 'e', 's', 't', '2', '0', '1', '7', '/', '0', '0', '0', '0', '0', '0', '0', '7', '3', '7', '2', '3', '.', 'j', 'p', 'g'],
        None)]

results = []
for img in outputs:
    # print(img[0])
    for i, person in enumerate(img[0]):
        # print(i)
        # print(person.shape) # (17,3)
        kps = person[:, :3]
        # print(kps.shape) # (17,3)
        kps = kps.reshape((-1)).round(3).tolist()
        # print(len(kps)) # 51
        kps = [round(k, 3) for k in kps]
        # print(img[1][0])
        score = round(float(img[1][i][5]), 3)
        id = ''
        # print(img[2])
        for key in img[2][19:31]:
            id = id + key
        # print(id)
        results.append({
            'category_id': int(1),
            'image_id': int(id),
            'keypoints': kps,
            'score': score
            })
print(results)

'''
with open(args.out,'w') as fid:
    json.dump(results, fid)
'''
####
