def map_data_2_id(iterable):
    """
    provides mapping to and from ids
    """

    id_2_data = {}
    data_2_id = {}

    for id, elem in enumerate(iterable):
        id_2_data[id] = elem

    for id, elem in enumerate(iterable):
        data_2_id[elem] = id

    return (id_2_data, data_2_id)


def one_hot(i, n):
    """
    create vector of size n with 1 at index i
    """
    x = jnp.zeros(n, dtype=int)
    print(f"i:\n{i}")
    print(f"type(i):\n{type(i)}")
    print(f"n:\n{n}")
    print(f"type(n):\n{type(n)}")
    x = x.at[i].set(1)  # jax.vmap(x.at[i].set(1), in_axes=(0, 0), out_axes=0)(i, x)

    print(f"i:\n{i}")
    print(f"type(x):\n{type(x)}")
    print(f"x:\n{x}")
    # array = x.at[i].set(1)
    # print(f"array:\n{array}")
    # x = x[i].at[i].set(1)
    # print(x)
    return x


def get_text(fname):
    with open(fname, "r") as reader:
        data = reader.read()
    return data


def prep_data(data):
    chars = list(set(data))
    vocab_size = len(chars)
    char_to_id, id_to_char = map_data_2_id(chars)
    char_to_id = {value: key for key, value in char_to_id.items()}
    # data converted to ids
    # data_id = [char_to_id[char] for char in data]
    data_id = [char_to_id[char] for char in data]
    return data_id, char_to_id, id_to_char


def encode(char):
    return one_hot(data_2_id[char], len(data_2_id))


def decode(predictions, id_2_data):
    return id_2_data[int(jnp.argmax(predictions))]
