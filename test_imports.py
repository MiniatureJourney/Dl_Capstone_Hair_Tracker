import sys, pandas as pd, numpy as np, csv
sys.path.insert(0,'src')

df = pd.read_csv('outputs/results/per_image_scores.csv')
print('=== Per-Image Score Statistics ===')
print('Total test images:', len(df))
print('Dice  mean:', round(df.dice.mean(),4), '| std:', round(df.dice.std(),4), '| min:', round(df.dice.min(),4), '| max:', round(df.dice.max(),4))
print('IoU   mean:', round(df.iou.mean(),4),  '| std:', round(df.iou.std(),4),  '| min:', round(df.iou.min(),4),  '| max:', round(df.iou.max(),4))
print('Prec  mean:', round(df.precision.mean(),4), '| std:', round(df.precision.std(),4))
print('Rec   mean:', round(df.recall.mean(),4),    '| std:', round(df.recall.std(),4))

print('')
print('=== Dice Score Distribution ===')
for lo, hi in [(0,0.5),(0.5,0.8),(0.8,0.9),(0.9,0.95),(0.95,1.01)]:
    cnt = ((df.dice >= lo) & (df.dice < hi)).sum()
    pct = cnt / len(df) * 100
    print(' ', lo, '-', hi, ':', cnt, 'images (', round(pct,1), '%)')

train_data = []
with open('outputs/training_log.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        train_data.append(row)

print('')
print('=== Training Convergence ===')
print('Total epochs:', len(train_data))
e1  = train_data[0]
e15 = train_data[-1]
print('Epoch 1  val_dice:', round(float(e1['val_dice']),4),  'train_loss:', round(float(e1['train_loss']),4))
print('Epoch 15 val_dice:', round(float(e15['val_dice']),4), 'train_loss:', round(float(e15['train_loss']),4))
print('Dice improvement :', round(float(e15['val_dice'])-float(e1['val_dice']),4))
print('Loss reduction   :', round(float(e1['train_loss'])-float(e15['train_loss']),4))
