import evaluate

pred_str=["我爱你"]
label_str=["我草似你"]

wer_metric = evaluate.load("wer")
wer = wer_metric.compute(predictions=pred_str, references=label_str)

print(type(wer))
