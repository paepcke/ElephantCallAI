import torch
import torch.nn as nn
import torch.nn.functional as F
# Not sure if we need this!
import parameters

def get_loss():
    '''
        Create the loss function based on the parameters file

        Return:
        loss - The loss function to use in training
        include_boundaries - boolean flag for whether the loss function needs
        to include boundary awareness to capture boundary uncertainty
    '''

    # Whether to include boundary uncertainty in the loss
    include_boundaries = False
    if parameters.LOSS.upper() == "CE":
        loss_func = nn.BCEWithLogitsLoss()
        print ("Using Binary Cross Entropy Loss")
    elif parameters.LOSS.upper() == "FOCAL":
        loss_func = FocalLoss(alpha=parameters.FOCAL_ALPHA, gamma=parameters.FOCAL_GAMMA)
        print ("Using Focal Loss with parameters alpha: {}, gamma: {}, pi: {}".format(parameters.FOCAL_ALPHA, parameters.FOCAL_GAMMA, parameters.FOCAL_WEIGHT_INIT))
    elif parameters.LOSS.upper() == "FOCAL_CHUNK":
        weight_func = None
        if parameters.CHUNK_WEIGHTING.upper() == "AVG":
            weight_func = avg_confidence_weighting
        elif parameters.CHUNK_WEIGHTING.upper() == "COUNT":
            weight_func = incorrect_count_weighting
        else:
            print ("Unknown Chunk Weighting for Focal Loss")
            # SHould probably break!
            return

        loss_func = ChunkFocalLoss(weight_func, alpha=parameters.FOCAL_ALPHA, gamma=parameters.FOCAL_GAMMA)
        print ("Using Chunk Based Focal Loss with weighting function: {}, alpha: {}, gamma: {}, pi: {}".format(
                parameters.CHUNK_WEIGHTING, parameters.FOCAL_ALPHA, 
                parameters.FOCAL_GAMMA, parameters.FOCAL_WEIGHT_INIT))
    elif parameters.LOSS.upper() == "BOUNDARY":
        include_boundaries = True

        if parameters.BOUNDARY_LOSS.upper() == "EQUAL":
            loss_func = BCE_Equal_Boundary_Loss()
        elif parameters.BOUNDARY_LOSS.upper() == "WEIGHT":
            loss_func = BCE_Weighted_Boundary_Loss(boundary_weight=parameters.BOUNDARY_WEIGHT)
        else:
            print ("Unknown Boundary Loss Type")
            return

        print("Using Boundary Enhanced Loss with Individual-Boundaries: {}, Boundary_Size: {}, Boundary Loss Type: {}, Weighting: {}".format(
                parameters.INDIVIDUAL_BOUNDARIES, parameters.BOUNDARY_FUDGE_FACTOR,
                parameters.BOUNDARY_LOSS, parameters.BOUNDARY_WEIGHT))
    else:
        print("Unknown Loss")
        return

    return loss_func, include_boundaries

# For the focal loss we just want to initialize the bias of the final layer to be
# log((1-pi)/pi) where pi = 0.01 is good
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # Calculate the standard BCE loss and then 
        # re-weight it by the term (a_t (1 - p_t)^gamma)
        # Let us see how the weighting term would actually apply here later!
        # Let us compare how this works an added alpha term
        bce_loss = self.bce(inputs, targets, reduction='none')
        # Get the actual values of pt = e ^ (log(pt)) from bce loss where we have -log(pt)
        pt = torch.exp(-bce_loss)

        #print (pt[targets==1])
        #print(targets)
        # Value of alpha for class 1 and 1 - alpha for class 0
        alpha = torch.tensor([1 - self.alpha, self.alpha]).to(parameters.device)
        # Select the appropriate alpha based on label y
        alpha_t = alpha[targets.data.view(-1).long()].view_as(targets)
        
        focal_loss = alpha_t * (1 - pt)**self.gamma * bce_loss

        # Let us look a bit into the amount of loss that goes into 
        # the negative vs. the postive examples
        """
        with torch.no_grad():
            print ("Num pos examples:", torch.sum(targets))
            print ("Confidence in correct class", pt)
            print ("Some of weight terms", (1 - pt)**self.gamma)
            loss_neg = torch.sum(focal_loss[targets == 0])
            loss_pos = torch.sum(focal_loss[targets==1])
            print ("Loss from negative examples:", loss_neg.item())
            print ("Loss from postive examples:", loss_pos.item())
            print ("Ratio of positive to negative:", torch.sum(targets).item() / targets.shape[0])
            print()
        """

        if self.reduce:
            return torch.mean(focal_loss)

        return focal_loss

class ChunkFocalLoss(nn.Module):
    def __init__(self, weight_func, alpha=0.25, gamma=2, batch_size=32, reduce=True):
        super(ChunkFocalLoss, self).__init__()
        self.weight_func = weight_func
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.reduce = reduce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
            Modified focal loss that re-weights at the chunk level
            rather than the individaul slice level. Namely, re-weight 
            the loss of each chunk based on its "difficulty" and how
            much attention we should pay to it in the future

            Parameters:
            inputs - [batch_size, chunk_length]
            targets - [batch_size, chunk_length]
        """
        bce_loss = self.bce(inputs, targets, reduction='none')
        # Small hack for now!
        bce_loss = bce_loss.view(self.batch_size, -1)

        pts = torch.exp(-bce_loss)

        # Determine the weighting we should pay to each individual
        # chunk in the batch
        chunk_weights = self.weight_func(pts, self.gamma) 

        # Calculate chunk based loss
        # Should we do mean or not here?
        # chunk_loss = [batch_size, 1] ??
        #chunk_loss = torch.mean(bce_loss, dim=1) # Why did I change to sum rather than mean???
        chunk_loss = torch.sum(bce_loss, dim=1)
        # Re-weight through focal loss scheme!
        # focal_loss = [batch_size, 1]
        focal_loss = chunk_weights * chunk_loss
        #focal_loss = (1 - chunk_weights)**self.gamma * chunk_loss

        if self.reduce:
            return torch.mean(focal_loss)

        return focal_loss

def avg_confidence_weighting(pts, weight):
    """
        Computes the weighting for each chunk based
        on the averge over (pts), where pt represents
        the confidence in the correct class for each slice.
        Then as in the focal loss paper re-weights by gamma

        Parameters:
        pts - [batch_size, chunk_length]: Gives confidence in prediction
        of the correct class for each slice
        weight - here weight represents gamma
    """
    return (1 - torch.mean(pts, dim=1)) ** weight

def incorrect_count_weighting(pts, weight):
    """
        Weight the difficulty of a given chunk by how many
        correct / incorrect slices are predicted (note this does
        not include confidence in such predictions).

        Parameters:
        pts - [batch_size, chunk_length]: Gives confidence in prediction
        of the correct class for each slice
        weight - here weight represents the denominator used to normalize
        the incorrect weightings
    """
    num_incorrect = torch.sum(pts < 0.5, dim=1).float()

    return (num_incorrect + 1) ** 2 / weight ** 2 # Think about maybe making sure these normalize to scale 0/1

class BCE_Equal_Boundary_Loss(nn.Module):
    def __init__(self):
        super(BCE_Equal_Boundary_Loss, self).__init__()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, boundary_masks):
        """
            Modified binary cross entropy loss that takes into account
            uncertainty around call boundaries. We augment the BCE loss
            by essentially allowing the model to be imprecise around the
            boarders. To do this we use the boundary_masks to change targets
            at the boundary to match the prediction. Therefore, we are essentially
            saying we do not care about being exact around the boundaries, we really
            care about the middle or meat of the call. Although this could allow the
            model to essentially predict anything around the boundaries, such as all
            0s or even non continuous 1s, the hope is that learning to pick up on some
            notion of the boundary being a set of 1s, then allows the model (especially
            in the case of lstm based model) to more easily predict the middle section
            as predicting 1s after already seeing some ones is easier than starting 
            from nothing.
        """
        
        # Fudge the ground truth predictions around the boundary. 
        # We assume the target labels are copies of dataset
        # Hacky for now but basically see the predicted 0/1
        ones = torch.ones_like(inputs)
        zeros = torch.zeros_like(inputs)
        predictions = torch.where(inputs > 0.5, ones, zeros).to(parameters.device).float()
        #predictions = torch.tensor(np.where(inputs > 0.5, ones, zeros)).to(parameters.device).float()
        #labels[boundary_masks] = 
        targets[boundary_masks] = predictions[boundary_masks]

        loss = self.loss_func(inputs, targets)

        return loss

class BCE_Weighted_Boundary_Loss(nn.Module):
    def __init__(self, boundary_weight):
        super(BCE_Weighted_Boundary_Loss, self).__init__()
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets, boundary_masks):
        """
            Modified binary cross entropy loss that takes into account
            uncertainty around call boundaries. We augment the BCE loss
            by essentially allowing the model to be imprecise around the
            boarders. To do this we use the boundary_masks to change targets
            at the boundary to match the prediction. Therefore, we are essentially
            saying we do not care about being exact around the boundaries, we really
            care about the middle or meat of the call. Although this could allow the
            model to essentially predict anything around the boundaries, such as all
            0s or even non continuous 1s, the hope is that learning to pick up on some
            notion of the boundary being a set of 1s, then allows the model (especially
            in the case of lstm based model) to more easily predict the middle section
            as predicting 1s after already seeing some ones is easier than starting 
            from nothing.
        """
        
        # Assign weighting to each slice, where the boundary slices
        # get weight (boundary_weight) and non_boundary slices get original 1 weighting
        loss_weights = (1 - boundary_masks.long()) + (boundary_masks.long() * self.boundary_weight)
        loss_weights = loss_weights.to(parameters.device).float()
        self.loss_func.weight = loss_weights

        loss = self.loss_func(inputs, targets)

        return loss

