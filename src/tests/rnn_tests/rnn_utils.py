import jax
import pytest
from jax import vmap

from src.data.process import moving_window, rolling_window

# --- Random Keys


@pytest.fixture
def key_generator():
    seed = 1701
    key = jax.random.PRNGKey(seed)
    num_copies = 23
    return jax.random.split(key, num=num_copies)


# ----------------------------------------------------------------
# ------------------ Single Cell Architectures Tests -------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------


@pytest.fixture
def one_to_many_single_cell_rnn_key(key_generator):
    return [key_generator[1], key_generator[2]]


@pytest.fixture
def one_to_many_single_cell_lstm_key(key_generator):
    return [key_generator[3], key_generator[4]]


@pytest.fixture
def one_to_many_single_cell_gru_key(key_generator):
    return [key_generator[5], key_generator[6]]


# -------------------------
# ------ many to one ------
# -------------------------


@pytest.fixture
def many_to_one_single_cell_rnn_key(key_generator):
    return [key_generator[7], key_generator[8]]


@pytest.fixture
def many_to_one_single_cell_lstm_key(key_generator):
    return [key_generator[9], key_generator[10]]


@pytest.fixture
def many_to_one_single_cell_gru_key(key_generator):
    return [key_generator[11], key_generator[12]]


# -------------------------
# ------ many to many -----
# -------------------------


@pytest.fixture
def many_to_many_single_cell_rnn_key(key_generator):
    return [key_generator[13], key_generator[14]]


@pytest.fixture
def many_to_many_single_cell_lstm_key(key_generator):
    return [key_generator[15], key_generator[16]]


@pytest.fixture
def many_to_many_single_cell_gru_key(key_generator):
    return [key_generator[17], key_generator[18]]


# ----------------------------------------------------------------
# ------------------ Single Cell Bidirectional Tests -------------
# ----------------------------------------------------------------


@pytest.fixture
def single_cell_birnn_key(key_generator):
    return [key_generator[19], key_generator[20]]


@pytest.fixture
def single_cell_bilstm_key(key_generator):
    return [key_generator[21], key_generator[22]]


@pytest.fixture
def single_cell_bigru_key(key_generator):
    return [key_generator[23], key_generator[24]]


# ----------------------------------------------------------------
# --------------------------- Deep RNN Tests ---------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------


@pytest.fixture
def one_to_many_deep_rnn_key(key_generator):
    return [key_generator[25], key_generator[26]]


@pytest.fixture
def one_to_many_deep_lstm_key(key_generator):
    return [key_generator[27], key_generator[28]]


@pytest.fixture
def one_to_many_deep_gru_key(key_generator):
    return [key_generator[29], key_generator[30]]


# -------------------------
# ------ many to one ------
# -------------------------


@pytest.fixture
def many_to_one_deep_rnn_key(key_generator):
    return [key_generator[31], key_generator[32]]


@pytest.fixture
def many_to_one_deep_lstm_key(key_generator):
    return [key_generator[33], key_generator[34]]


@pytest.fixture
def many_to_one_deep_gru_key(key_generator):
    return [key_generator[35], key_generator[36]]


# -------------------------
# ------ many to many -----
# -------------------------


@pytest.fixture
def many_to_many_deep_rnn_key(key_generator):
    return [key_generator[37], key_generator[38]]


@pytest.fixture
def many_to_many_deep_lstm_key(key_generator):
    return [key_generator[39], key_generator[40]]


@pytest.fixture
def many_to_many_deep_gru_key(key_generator):
    return [key_generator[41], key_generator[42]]


# ----------------------------------------------------------------
# ------------------------- Deep Bidirectional -------------------
# ----------------------------------------------------------------


@pytest.fixture
def deep_birnn_key(key_generator):
    return [key_generator[43], key_generator[44]]


@pytest.fixture
def deep_bilstm_key(key_generator):
    return [key_generator[45], key_generator[46]]


@pytest.fixture
def deep_bigru_key(key_generator):
    return [key_generator[47], key_generator[48]]


# ----------------------------------------------------------------
# -------------------- Single Cell HiPPO LSTM --------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def one_to_many_single_hippo_legs_lstm_key(key_generator):
    return [key_generator[49], key_generator[50]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def one_to_many_single_hippo_legt_lstm_key(key_generator):
    return [key_generator[51], key_generator[52]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def one_to_many_single_hippo_lmu_lstm_key(key_generator):
    return [key_generator[53], key_generator[54]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def one_to_many_single_hippo_lagt_lstm_key(key_generator):
    return [key_generator[55], key_generator[56]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def one_to_many_single_hippo_fru_lstm_key(key_generator):
    return [key_generator[57], key_generator[58]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def one_to_many_single_hippo_fout_lstm_key(key_generator):
    return [key_generator[59], key_generator[60]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def one_to_many_single_hippo_foud_lstm_key(key_generator):
    return [key_generator[61], key_generator[62]]


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_one_single_hippo_legs_lstm_key(key_generator):
    return [key_generator[63], key_generator[64]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_one_single_hippo_legt_lstm_key(key_generator):
    return [key_generator[65], key_generator[66]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_one_single_hippo_lmu_lstm_key(key_generator):
    return [key_generator[67], key_generator[68]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_one_single_hippo_lagt_lstm_key(key_generator):
    return [key_generator[69], key_generator[70]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_one_single_hippo_fru_lstm_key(key_generator):
    return [key_generator[71], key_generator[72]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_one_single_hippo_fout_lstm_key(key_generator):
    return [key_generator[73], key_generator[74]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_one_single_hippo_foud_lstm_key(key_generator):
    return [key_generator[75], key_generator[76]]


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_many_single_hippo_legs_lstm_key(key_generator):
    return [key_generator[77], key_generator[78]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_many_single_hippo_legt_lstm_key(key_generator):
    return [key_generator[79], key_generator[80]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_many_single_hippo_lmu_lstm_key(key_generator):
    return [key_generator[81], key_generator[82]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_many_single_hippo_lagt_lstm_key(key_generator):
    return [key_generator[83], key_generator[84]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_many_single_hippo_fru_lstm_key(key_generator):
    return [key_generator[85], key_generator[86]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_many_single_hippo_fout_lstm_key(key_generator):
    return [key_generator[87], key_generator[88]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_many_single_hippo_foud_lstm_key(key_generator):
    return [key_generator[89], key_generator[90]]


# ----------------------------------------------------------------
# -------------------- Single Cell HiPPO GRU ---------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def one_to_many_single_hippo_legs_gru_key(key_generator):
    return [key_generator[91], key_generator[92]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def one_to_many_single_hippo_legt_gru_key(key_generator):
    return [key_generator[93], key_generator[94]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def one_to_many_single_hippo_lmu_gru_key(key_generator):
    return [key_generator[95], key_generator[96]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def one_to_many_single_hippo_lagt_gru_key(key_generator):
    return [key_generator[97], key_generator[98]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def one_to_many_single_hippo_fru_gru_key(key_generator):
    return [key_generator[99], key_generator[100]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def one_to_many_single_hippo_fout_gru_key(key_generator):
    return [key_generator[101], key_generator[102]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def one_to_many_single_hippo_foud_gru_key(key_generator):
    return [key_generator[103], key_generator[104]]


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_one_single_hippo_legs_gru_key(key_generator):
    return [key_generator[105], key_generator[106]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_one_single_hippo_legt_gru_key(key_generator):
    return [key_generator[107], key_generator[108]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_one_single_hippo_lmu_gru_key(key_generator):
    return [key_generator[109], key_generator[110]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_one_single_hippo_lagt_gru_key(key_generator):
    return [key_generator[111], key_generator[112]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_one_single_hippo_fru_gru_key(key_generator):
    return [key_generator[113], key_generator[114]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_one_single_hippo_fout_gru_key(key_generator):
    return [key_generator[115], key_generator[116]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_one_single_hippo_foud_gru_key(key_generator):
    return [key_generator[117], key_generator[118]]


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_many_single_hippo_legs_gru_key(key_generator):
    return [key_generator[119], key_generator[120]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_many_single_hippo_legt_gru_key(key_generator):
    return [key_generator[121], key_generator[122]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_many_single_hippo_lmu_gru_key(key_generator):
    return [key_generator[123], key_generator[124]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_many_single_hippo_lagt_gru_key(key_generator):
    return [key_generator[125], key_generator[126]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_many_single_hippo_fru_gru_key(key_generator):
    return [key_generator[127], key_generator[128]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_many_single_hippo_fout_gru_key(key_generator):
    return [key_generator[129], key_generator[130]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_many_single_hippo_foud_gru_key(key_generator):
    return [key_generator[131], key_generator[132]]


# ----------------------------------------------------------------
# ------------ Single Cell Bidirectional HiPPO LSTM --------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def single_cell_hippo_legs_bilstm_key(key_generator):
    return [key_generator[133], key_generator[134]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def single_cell_hippo_legt_bilstm_key(key_generator):
    return [key_generator[135], key_generator[136]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def single_cell_hippo_lmu_bilstm_key(key_generator):
    return [key_generator[137], key_generator[138]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def single_cell_hippo_lagt_bilstm_key(key_generator):
    return [key_generator[139], key_generator[140]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def single_cell_hippo_fru_bilstm_key(key_generator):
    return [key_generator[141], key_generator[142]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def single_cell_hippo_fout_bilstm_key(key_generator):
    return [key_generator[143], key_generator[144]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def single_cell_hippo_foud_bilstm_key(key_generator):
    return [key_generator[145], key_generator[146]]


# ----------------------------------------------------------------
# ------------ Single Cell Bidirectional HiPPO GRU ---------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def single_cell_hippo_legs_bigru_key(key_generator):
    return [key_generator[147], key_generator[148]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def single_cell_hippo_legt_bigru_key(key_generator):
    return [key_generator[149], key_generator[150]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def single_cell_hippo_lmu_bigru_key(key_generator):
    return [key_generator[151], key_generator[152]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def single_cell_hippo_lagt_bigru_key(key_generator):
    return [key_generator[153], key_generator[154]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def single_cell_hippo_fru_bigru_key(key_generator):
    return [key_generator[155], key_generator[156]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def single_cell_hippo_fout_bigru_key(key_generator):
    return [key_generator[157], key_generator[158]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def single_cell_hippo_foud_bigru_key(key_generator):
    return [key_generator[159], key_generator[160]]


# ----------------------------------------------------------------
# ------------------------ Deep HiPPO LSTM -----------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_legs_lstm_key(key_generator):
    return [key_generator[161], key_generator[162]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_legt_lstm_key(key_generator):
    return [key_generator[163], key_generator[164]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_lmu_lstm_key(key_generator):
    return [key_generator[165], key_generator[166]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_lagt_lstm_key(key_generator):
    return [key_generator[167], key_generator[168]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_fru_lstm_key(key_generator):
    return [key_generator[169], key_generator[170]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def one_to_many_deep_hippo_fout_lstm_key(key_generator):
    return [key_generator[171], key_generator[172]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def one_to_many_deep_hippo_foud_lstm_key(key_generator):
    return [key_generator[173], key_generator[174]]


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_legs_lstm_key(key_generator):
    return [key_generator[175], key_generator[176]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_legt_lstm_key(key_generator):
    return [key_generator[177], key_generator[178]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_lmu_lstm_key(key_generator):
    return [key_generator[179], key_generator[180]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_lagt_lstm_key(key_generator):
    return [key_generator[181], key_generator[182]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_fru_lstm_key(key_generator):
    return [key_generator[183], key_generator[184]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_one_deep_hippo_fout_lstm_key(key_generator):
    return [key_generator[185], key_generator[186]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_one_deep_hippo_foud_lstm_key(key_generator):
    return [key_generator[187], key_generator[188]]


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_legs_lstm_key(key_generator):
    return [key_generator[189], key_generator[190]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_legt_lstm_key(key_generator):
    return [key_generator[191], key_generator[192]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_lmu_lstm_key(key_generator):
    return [key_generator[193], key_generator[194]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_lagt_lstm_key(key_generator):
    return [key_generator[195], key_generator[196]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_fru_lstm_key(key_generator):
    return [key_generator[197], key_generator[198]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_many_deep_hippo_fout_lstm_key(key_generator):
    return [key_generator[199], key_generator[200]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_many_deep_hippo_foud_lstm_key(key_generator):
    return [key_generator[201], key_generator[202]]


# ----------------------------------------------------------------
# ------------------------ Deep HiPPO GRU ------------------------
# ----------------------------------------------------------------

# -------------------------
# ------ one to many ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_legs_gru_key(key_generator):
    return [key_generator[203], key_generator[204]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_legt_gru_key(key_generator):
    return [key_generator[205], key_generator[206]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_lmu_gru_key(key_generator):
    return [key_generator[207], key_generator[208]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_lagt_gru_key(key_generator):
    return [key_generator[209], key_generator[210]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def one_to_many_deep_hippo_fru_gru_key(key_generator):
    return [key_generator[211], key_generator[212]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def one_to_many_deep_hippo_fout_gru_key(key_generator):
    return [key_generator[213], key_generator[214]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def one_to_many_deep_hippo_foud_gru_key(key_generator):
    return [key_generator[215], key_generator[216]]


# -------------------------
# ------ many to one ------
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_legs_gru_key(key_generator):
    return [key_generator[217], key_generator[218]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_legt_gru_key(key_generator):
    return [key_generator[219], key_generator[220]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_lmu_gru_key(key_generator):
    return [key_generator[221], key_generator[222]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_lagt_gru_key(key_generator):
    return [key_generator[223], key_generator[224]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_one_deep_hippo_fru_gru_key(key_generator):
    return [key_generator[225], key_generator[226]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_one_deep_hippo_fout_gru_key(key_generator):
    return [key_generator[227], key_generator[228]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_one_deep_hippo_foud_gru_key(key_generator):
    return [key_generator[229], key_generator[230]]


# -------------------------
# ------ many to many -----
# -------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_legs_gru_key(key_generator):
    return [key_generator[231], key_generator[232]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_legt_gru_key(key_generator):
    return [key_generator[233], key_generator[234]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_lmu_gru_key(key_generator):
    return [key_generator[235], key_generator[236]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_lagt_gru_key(key_generator):
    return [key_generator[237], key_generator[238]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def many_to_many_deep_hippo_fru_gru_key(key_generator):
    return [key_generator[239], key_generator[240]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def many_to_many_deep_hippo_fout_gru_key(key_generator):
    return [key_generator[241], key_generator[242]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def many_to_many_deep_hippo_foud_gru_key(key_generator):
    return [key_generator[243], key_generator[244]]


# ----------------------------------------------------------------
# ---------------- Deep Bidirectional HiPPO LSTM -----------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def deep_hippo_legs_bilstm_key(key_generator):
    return [key_generator[245], key_generator[246]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def deep_hippo_legt_bilstm_key(key_generator):
    return [key_generator[247], key_generator[248]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def deep_hippo_lmu_bilstm_key(key_generator):
    return [key_generator[249], key_generator[250]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def deep_hippo_lagt_bilstm_key(key_generator):
    return [key_generator[251], key_generator[252]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def deep_hippo_fru_bilstm_key(key_generator):
    return [key_generator[253], key_generator[254]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def deep_hippo_fout_bilstm_key(key_generator):
    return [key_generator[255], key_generator[256]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def deep_hippo_foud_bilstm_key(key_generator):
    return [key_generator[257], key_generator[258]]


# ----------------------------------------------------------------
# ----------------- Deep Bidirectional HiPPO GRU -----------------
# ----------------------------------------------------------------

# ----------
# -- legs --
# ----------


@pytest.fixture
def deep_hippo_legs_bigru_key(key_generator):
    return [key_generator[259], key_generator[260]]


# ----------
# -- legt --
# ----------


@pytest.fixture
def deep_hippo_legt_bigru_key(key_generator):
    return [key_generator[261], key_generator[262]]


# ----------
# -- lmu --
# ----------


@pytest.fixture
def deep_hippo_lmu_bigru_key(key_generator):
    return [key_generator[263], key_generator[264]]


# ----------
# -- lagt --
# ----------


@pytest.fixture
def deep_hippo_lagt_bigru_key(key_generator):
    return [key_generator[265], key_generator[266]]


# ----------
# --- fru --
# ----------


@pytest.fixture
def deep_hippo_fru_bigru_key(key_generator):
    return [key_generator[267], key_generator[268]]


# ------------
# --- fout ---
# ------------


@pytest.fixture
def deep_hippo_fout_bigru_key(key_generator):
    return [key_generator[269], key_generator[270]]


# ------------
# --- foud ---
# ------------


@pytest.fixture
def deep_hippo_foud_bigru_key(key_generator):
    return [key_generator[271], key_generator[272]]


# ----------------------------------------------------------------
# ------------------------ Sequence Inputs -----------------------
# ----------------------------------------------------------------


@pytest.fixture
def random_16_input(key_generator):
    batch_size = 16
    data_size = 28 * 28
    input_size = 28
    x = jax.random.randint(key_generator[273], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)


@pytest.fixture
def random_32_input(key_generator):
    batch_size = 32
    data_size = 28 * 28
    input_size = 28
    x = jax.random.randint(key_generator[274], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)


@pytest.fixture
def random_64_input(key_generator):
    batch_size = 64
    data_size = 28 * 28
    input_size = 28
    x = jax.random.randint(key_generator[275], (batch_size, data_size), 0, 255)
    return vmap(moving_window, in_axes=(0, None))(x, input_size)
