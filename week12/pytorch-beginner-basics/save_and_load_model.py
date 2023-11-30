import torch
import torchvision.models as models

print('Save/Load Model Weights Only, using state_dict')
# PyTorch models store the learned parameters in an internal state dictionary, called state_dict. These can be persisted via the torch.save method:
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method:
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth'))
model.eval() # be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.

# Save/Load Entire Model
print('Save/Load Entire Model')
torch.save(model, 'model.pth')
model = torch.load('model.pth')
