from classify.predictor import MobileNetClassify, MobileNetClassifyOnnx
import time
import fire

def predict(impath, weight='weights/mobilenet_v2-09_acc0.9989_loss0.0051.pth',
            idx2class="weights/idcard_classname.json", topk=1, mode='torch'):
    if mode=="torch":
        model = MobileNetClassify(weight=weight, topk=topk, idx2class=idx2class)
    elif mode=='onnx':
        model = MobileNetClassifyOnnx(weight=weight, topk=topk, idx2class=idx2class)
    else:
        raise Exception("Only torch and onnx mode are supported!")
    
    start_time = time.time()
    
    result = model.predict(impath)
    
    total_time =time.time() - start_time
    unit = "s"
    if total_time<1:
        total_time = float(total_time * 1000)
        unit="ms"
        
    print(f'Result : {result}')
    print(f'Execution Time: {total_time:.0f} {unit}')
    
    
    
if __name__ == '__main__':
    fire.Fire(predict)