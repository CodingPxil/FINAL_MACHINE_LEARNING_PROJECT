def train():
  log = CSVLogger('lightning_logs', name='MLP')
  early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
  trainer = lightning.Trainer(max_epochs=epochs, logger=log, callbacks=[early_stopping])
  model = PneumoniaModel(3, learning_rate)
  train, val, test, classes = get_dataloaders(path + "/chest_xray")
  trainer.fit(model, train, val)
  
  return model
