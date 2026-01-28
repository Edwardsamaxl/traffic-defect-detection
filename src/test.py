from ultralytics.nn import modules
print(modules.EMA)         # <class 'ultralytics.nn.modules.ema.EMA'>
print(modules.__dict__['EMA'])  # 同样可以找到
