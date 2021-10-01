from classify.predictor import MobileNetClassify
import time
import fire

def predict(impath, weight='weights/mobilenet_v2-09_acc0.9989_loss0.0051.pth',topk=1):
    model = MobileNetClassify(weight=weight,topk=topk)
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