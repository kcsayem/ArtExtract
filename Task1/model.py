import torch
import torch.nn as nn
import torchvision.models as models


class CNN_RNN_Model(nn.Module):
    def __init__(self, num_classes_style, num_classes_artist, num_classes_genre, cnn_hidden_size, rnn_hidden_size):
        super(CNN_RNN_Model, self).__init__()
        self.cnn = models.resnet50(pretrained=True)  # You can choose any pretrained CNN model
        in_features = self.cnn.fc.in_features
        self.cnn.fc =  nn.Identity()

        self.rnn = nn.LSTM(input_size=2048, hidden_size=rnn_hidden_size, num_layers=1, batch_first=True)

        self.fc_style = nn.Linear(rnn_hidden_size, num_classes_style)
        self.fc_artist = nn.Linear(rnn_hidden_size, num_classes_artist)
        self.fc_genre = nn.Linear(rnn_hidden_size, num_classes_genre)

    def forward(self, x):
        # CNN feature extraction
        # with torch.no_grad():
        cnn_features = self.cnn(x)

        # RNN processing
        rnn_output, _ = self.rnn(cnn_features)

        # Classification for style, artist, and genre
        output_style = self.fc_style(rnn_output)  # Only consider the last RNN output
        output_artist = self.fc_artist(rnn_output)
        output_genre = self.fc_genre(rnn_output)

        return output_style, output_artist, output_genre

    @property
    def device(self):
        return next(self.parameters()).device


if __name__ == "__main__":
    # Example usage:
    num_classes_style = 1  # Number of style classes
    num_classes_artist = 1  # Number of artist classes
    num_classes_genre = 1  # Number of genre classes
    cnn_hidden_size = 2048  # Output size of the CNN feature extractor
    rnn_hidden_size = 512  # Hidden size of the RNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model
    model = CNN_RNN_Model(num_classes_style, num_classes_artist, num_classes_genre, cnn_hidden_size, rnn_hidden_size).to(device=device)

    # Example input tensor
    batch_size = 32
    input_tensor = torch.randn(batch_size, 3, 224, 224, device=device)  # Assuming input images are 224x224 RGB

    # Forward pass
    from tqdm import tqdm
    for i in tqdm(range(100)):
        output_style, output_artist, output_genre = model(input_tensor)

    print("Output style shape:", output_style.shape)  # Shape: (batch_size, num_classes_style)
    print("Output artist shape:", output_artist.shape)  # Shape: (batch_size, num_classes_artist)
    print("Output genre shape:", output_genre.shape)  # Shape: (batch_size, num_classes_genre)
