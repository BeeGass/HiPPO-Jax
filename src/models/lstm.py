import math
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """

    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # define properties
        # (techincally unnecessary)

        self.W_gx = None
        self.W_gh = None

        self.W_ix = None
        self.W_ih = None

        self.W_fx = None
        self.W_fh = None

        self.W_ox = None
        self.W_oh = None

        self.b_g = None
        self.b_i = None
        self.b_f = None

        self.c_t = None
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        k = float(1.0 / torch.sqrt(torch.Tensor([self.hidden_dim])))

        # input modulation params
        self.W_gx = nn.Parameter(
            torch.FloatTensor(self.embed_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.W_gh = nn.Parameter(
            torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.b_g = nn.Parameter(
            torch.FloatTensor(self.hidden_dim).uniform_(-1 * k, 1 * k)
            # +
            # torch.ones(self.hidden_dim)
            ,
            requires_grad=True,
        )

        # input gate params
        self.W_ix = nn.Parameter(
            torch.FloatTensor(self.embed_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.W_ih = nn.Parameter(
            torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.b_i = nn.Parameter(
            torch.FloatTensor(self.hidden_dim).uniform_(-1 * k, 1 * k)
            # +
            # torch.ones(self.hidden_dim)
            ,
            requires_grad=True,
        )

        # forget gate params

        self.W_fx = nn.Parameter(
            torch.FloatTensor(self.embed_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.W_fh = nn.Parameter(
            torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.b_f = nn.Parameter(
            torch.FloatTensor(self.hidden_dim).uniform_(-1 * k, 1 * k)
            + torch.ones(self.hidden_dim),
            requires_grad=True,
        )

        # output gate params

        self.W_ox = nn.Parameter(
            torch.FloatTensor(self.embed_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.W_oh = nn.Parameter(
            torch.FloatTensor(self.hidden_dim, self.hidden_dim).uniform_(-1 * k, 1 * k),
            requires_grad=True,
        )

        self.b_o = nn.Parameter(
            torch.FloatTensor(self.hidden_dim).uniform_(-1 * k, 1 * k)
            # +
            # torch.ones(self.hidden_dim)
            ,
            requires_grad=True,
        )

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds, h_t=None, c_t=None):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, hidden dimension].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # initialize hidden state if not given
        if h_t is None:
            h_t = torch.zeros(embeds.size(1), self.hidden_dim).to(embeds.device)

        # initialize cell state if not given
        if c_t is None:
            c_t = torch.zeros(embeds.size(1), self.hidden_dim).to(embeds.device)

        # initialize output
        output = torch.zeros(embeds.size(0), embeds.size(1), self.hidden_dim).to(
            embeds.device
        )

        # iterate over time steps

        for t in range(embeds.size(0)):
            # calculate input gate
            i_t = torch.sigmoid(
                torch.matmul(embeds[t], self.W_ix)
                + torch.matmul(h_t, self.W_ih)
                + self.b_i
            )

            # calculate forget gate
            f_t = torch.sigmoid(
                torch.matmul(embeds[t], self.W_fx)
                + torch.matmul(h_t, self.W_fh)
                + self.b_f
            )

            # calculate input modulation gate
            g_t = torch.tanh(
                torch.matmul(embeds[t], self.W_gx)
                + torch.matmul(h_t, self.W_gh)
                + self.b_g
            )

            # calculate output gate
            o_t = torch.sigmoid(
                torch.matmul(embeds[t], self.W_ox)
                + torch.matmul(h_t, self.W_oh)
                + self.b_o
            )

            # calculate cell state
            c_t = f_t * c_t + i_t * g_t

            # calculate hidden state
            h_t = o_t * torch.tanh(c_t)

            # store output
            output[t] = h_t

        # We save the hidden state for sampling
        self.c_t = c_t.clone()

        return output
