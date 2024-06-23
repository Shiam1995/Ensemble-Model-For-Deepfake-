import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches


def draw_neural_network(input_size, hidden_sizes, output_size):
    G = nx.DiGraph()

    # Input layer
    input_nodes = ['Input ' + str(i + 1) for i in range(input_size)]
    for i in range(input_size):
        G.add_node(input_nodes[i], pos=(0, i))

    # Hidden layers
    hidden_nodes = []
    for layer_index, hidden_size in enumerate(hidden_sizes):
        current_hidden_nodes = ['Hidden L{} N{}'.format(layer_index + 1, i + 1) for i in range(hidden_size)]
        hidden_nodes.append(current_hidden_nodes)
        for i in range(hidden_size):
            G.add_node(current_hidden_nodes[i], pos=(layer_index + 1, i + (input_size - hidden_size) // 2))

    # Output layer
    output_nodes = ['Output ' + str(i + 1) for i in range(output_size)]
    for i in range(output_size):
        G.add_node(output_nodes[i], pos=(len(hidden_sizes) + 1, i + (input_size - output_size) // 2))

    # Add edges with weights
    edges = []
    # Edges from input to first hidden layer
    for i in range(input_size):
        for j in range(len(hidden_nodes[0])):
            edges.append((input_nodes[i], hidden_nodes[0][j]))

    # Edges between hidden layers
    for layer_index in range(len(hidden_sizes) - 1):
        for i in range(len(hidden_nodes[layer_index])):
            for j in range(len(hidden_nodes[layer_index + 1])):
                edges.append((hidden_nodes[layer_index][i], hidden_nodes[layer_index + 1][j]))

    # Edges from last hidden layer to output layer
    for i in range(len(hidden_nodes[-1])):
        for j in range(output_size):
            edges.append((hidden_nodes[-1][i], output_nodes[j]))

    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(16, 12))

    # Colors for nodes
    input_color = 'green'
    output_color = 'yellow'
    hidden_colors = ['lightgreen', 'limegreen', 'forestgreen', 'seagreen', 'mediumseagreen', 'darkgreen',
                     'yellowgreen'][:len(hidden_sizes)]

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color=input_color, node_size=3000)
    for layer_index, current_hidden_nodes in enumerate(hidden_nodes):
        nx.draw_networkx_nodes(G, pos, nodelist=current_hidden_nodes, node_color=hidden_colors[layer_index],
                               node_size=3000)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color=output_color, node_size=3000)

    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='-|>', arrowsize=20)

    # Add legend
    input_patch = mpatches.Patch(color=input_color, label='Input Layer')
    hidden_patches = [mpatches.Patch(color=color, label=f'Hidden Layer {i + 1}') for i, color in
                      enumerate(hidden_colors)]
    output_patch = mpatches.Patch(color=output_color, label='Output Layer')
    plt.legend(handles=[input_patch] + hidden_patches + [output_patch])

    # Title and text box
    plt.title('Neural Network Architecture')
    plt.text(-0.5, max(input_size, max(hidden_sizes), output_size) + 1,
             "Input Layer (green): Input nodes\nHidden Layers (shades of green): Hidden nodes\nOutput Layer (yellow): Output nodes",
             fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.show()


# Define the size of each layer
input_layer_size = 10
hidden_layer_sizes = [6, 7, 6, 7, 6, 7]  # List of hidden layer sizes
output_layer_size = 3

# Draw the neural network
draw_neural_network(input_layer_size, hidden_layer_sizes, output_layer_size)