import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.models import basic
from tensor2tensor.models import lstm
from tensor2tensor.models.research import vqa_attention


def basic_fc_relu(hparams, x, name="basic_fc_relu"):
    with tf.variable_scope(name):
        shape = common_layers.shape_list(x)
        x = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
        for i in range(hparams.num_hidden_layers):
            x = tf.layers.dense(x, hparams.hidden_size, name="layer_%d" % i)
            x = tf.nn.dropout(x, keep_prob=1.0 - hparams.dropout)
            x = tf.nn.relu(x)
        return tf.expand_dims(tf.expand_dims(x, axis=1), axis=1)  # 4D For T2T.


@registry.register_model
class ImageOnly(t2t_model.T2TModel):
    """Use image as inputs."""

    def body(self, features):
        hparams = self.hparams
        image_feat = features["image"]
        outputs = basic_fc_relu(hparams, image_feat)
        return outputs


@registry.register_model
class ImageAndPrice(t2t_model.T2TModel):
    """Use image and price as inputs."""

    def body(self, features):
        hparams = self.hparams
        image_feat = features["image"]
        price = tf.math.log(tf.to_float(features["price"]) + 1e-7)

        print("*** Image and Price ***")
        print("- image: {}, {}".format(image_feat.shape, image_feat.dtype))
        print("- price: {}, {}".format(price.shape, price.dtype))

        price = common_layers.expand_squeeze_to_nd(
            price, len(image_feat.shape), expand_dim=1)
        inputs = tf.concat([image_feat, price], axis=-1)

        print("- inputs: {}, {}".format(inputs.shape, inputs.dtype))

        outputs = basic_fc_relu(hparams, inputs)
        return outputs


@registry.register_hparams
def image_only_base():
    """Big fully connected image model."""
    hparams = basic.basic_fc_small()
    hparams.hidden_size = 1024
    hparams.batch_size = 1024
    return hparams


@registry.register_hparams
def image_only_small():
    """Small fully connected image model."""
    hparams = image_only_base()
    hparams.hidden_size = 256
    hparams.batch_size = 16384
    return hparams


@registry.register_hparams
def image_only_tall():
    """Small fully connected image model."""
    hparams = image_only_base()
    hparams.batch_size = 16384
    hparams.num_hidden_layers = 4
    return hparams


@registry.register_model
class ImageLSTMConcat(lstm.LSTMSeq2seqAttentionBidirectionalEncoder):
    """Use image as inputs."""

    def body(self, features):
        base_outputs = super().body(features)

        hparams = self.hparams
        x = features["image"]
        image_outputs = basic_fc_relu(hparams, x)

        outputs = tf.concat([base_outputs, image_outputs], axis=-1)
        return outputs


def attn(image_feat, query, hparams, name="attn"):
    """Attention on image feature with question as query."""
    print("*** Attention ***")
    print("- image_feat: {}, {}".format(image_feat.shape, image_feat.dtype))
    print("- query: {}, {}".format(query.shape, query.dtype))
    with tf.variable_scope(name, "attn", values=[image_feat, query]):
        attn_dim = hparams.attn_dim
        num_glimps = hparams.num_glimps
        num_channels = common_layers.shape_list(image_feat)[-1]
        if len(common_layers.shape_list(image_feat)) == 4:
            image_feat = common_layers.flatten4d3d(image_feat)
        if len(common_layers.shape_list(query)) == 4:
            query = common_layers.flatten4d3d(query)
        if len(common_layers.shape_list(query)) == 2:
            query = tf.expand_dims(query, 1)
        image_proj = common_attention.compute_attention_component(
            image_feat, attn_dim, name="image_proj")
        query_proj = common_attention.compute_attention_component(
            query, attn_dim, name="query_proj")
        h = tf.nn.relu(image_proj + query_proj)
        h_proj = common_attention.compute_attention_component(
            h, num_glimps, name="h_proj")
        p = tf.nn.softmax(h_proj, axis=1)
        image_ave = tf.matmul(image_feat, p, transpose_a=True)
        image_ave = tf.reshape(image_ave, [-1, num_channels * num_glimps])

        return image_ave


def mlp(feature, hparams, name="mlp"):
    """Multi layer perceptron with dropout and relu activation."""
    print("*** MLP ***")
    print("- feature: {}, {}".format(feature.shape, feature.dtype))
    with tf.variable_scope(name, "mlp", values=[feature]):
        num_mlp_layers = hparams.num_mlp_layers
        mlp_dim = hparams.mlp_dim
        for _ in range(num_mlp_layers):
            feature = common_layers.dense(feature, mlp_dim, activation=tf.nn.relu)
            feature = tf.nn.dropout(feature, keep_prob=1. - hparams.dropout)
        return feature


@registry.register_model
class ImageFcAttention(lstm.LSTMSeq2seqAttentionBidirectionalEncoder):
    """Use image as inputs."""

    def body(self, features):
        query = super().body(features)

        hparams = self.hparams
        image_feat = features["image"]

        # image_feat 에 layer 정규화 및 dropout 적용
        image_feat = common_layers.l2_norm(image_feat)

        # Fully connected layers for the image feature
        image_feat = basic_fc_relu(hparams, image_feat)

        # Attention on image feature with question as query
        image_ave = attn(image_feat, query, hparams)
        utils.collect_named_outputs("norms", "image_ave", tf.norm(image_ave, axis=-1))

        query = tf.squeeze(query, axis=[1, 2])

        image_text = tf.concat([image_ave, query], axis=1)
        utils.collect_named_outputs("norms", "image_text", tf.norm(image_text, axis=-1))
        image_text = tf.nn.dropout(image_text, 1. - hparams.dropout)

        # dropout 및 relu 가 있는 multi layer perceptron
        output = mlp(image_text, hparams)
        utils.collect_named_outputs("norms", "output", tf.norm(output, axis=-1))

        # Expand dimension 1 and 2
        return tf.expand_dims(tf.expand_dims(output, axis=1), axis=2)


@registry.register_model
class ImageSelfAttention(lstm.LSTMSeq2seqAttentionBidirectionalEncoder):
    """Use image as inputs."""

    def body(self, features):
        query = super().body(features)

        hparams = self.hparams
        image_feat = features["image"]

        image_feat = common_layers.flatten4d3d(image_feat)
        image_feat = tf.nn.dropout(image_feat, keep_prob=1. - hparams.dropout)

        image_feat = vqa_attention.image_encoder(image_feat, hparams)
        utils.collect_named_outputs("norms", "image_feat_encoded",
                                    tf.norm(image_feat, axis=-1))

        # image_feat 에 layer 정규화 및 dropout 적용
        image_feat = common_layers.l2_norm(image_feat)

        utils.collect_named_outputs("norms", "image_feat_encoded_l2",
                                    tf.norm(image_feat, axis=-1))

        # Attention on image feature with question as query
        image_ave = attn(image_feat, query, hparams)
        utils.collect_named_outputs("norms", "image_ave", tf.norm(image_ave, axis=-1))

        query = tf.squeeze(query, axis=[1, 2])

        image_text = tf.concat([image_ave, query], axis=1)
        utils.collect_named_outputs("norms", "image_text", tf.norm(image_text, axis=-1))
        image_text = tf.nn.dropout(image_text, 1. - hparams.dropout)

        # dropout 및 relu 가 있는 multi layer perceptron
        output = mlp(image_text, hparams)
        utils.collect_named_outputs("norms", "output", tf.norm(output, axis=-1))

        # Expand dimension 1 and 2
        return tf.expand_dims(tf.expand_dims(output, axis=1), axis=2)


@registry.register_hparams
def image_attention_base():
    """Image attention baseline hparams."""
    hparams = lstm_base_batch_4k_hidden_1k()

    # Attention
    hparams.add_hparam("attn_dim", 512)
    hparams.add_hparam("num_glimps", 2)

    # MLP
    hparams.add_hparam("num_mlp_layers", 1)
    hparams.add_hparam("mlp_dim", 1024)

    return hparams


@registry.register_hparams
def lstm_base_batch_8k_hidden_1k():
    """Set of hyperparameters."""
    hparams = lstm.lstm_attention()

    hparams.batch_size = 8192
    hparams.hidden_size = 1024
    hparams.attention_layer_size = 1024

    return hparams


@registry.register_hparams
def lstm_base_batch_4k_hidden_1k():
    """Set of hyperparameters."""
    hparams = lstm_base_batch_8k_hidden_1k()

    hparams.batch_size = 4096

    return hparams


@registry.register_hparams
def lstm_base_batch_16k_hidden_1k():
    """Set of hyperparameters."""
    hparams = lstm_base_batch_8k_hidden_1k()

    hparams.batch_size = 16384

    return hparams


@registry.register_hparams
def lstm_big():
    """Baseline with different learning rate schedule"""
    hparams = lstm_base_batch_16k_hidden_1k()

    hparams.optimizer = "AdamW"
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.999
    hparams.optimizer_adam_epsilon = 1e-8

    hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
    hparams.learning_rate_constant = 2e-4
    hparams.learning_rate_warmup_steps = 8000
    hparams.learning_rate_decay_steps = 80000

    return hparams


@registry.register_hparams
def lstm_big_multistep8():
    """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
    hparams = lstm_big()

    hparams.optimizer = "MultistepAdam"
    hparams.optimizer_multistep_accumulate_steps = 8

    return hparams


@registry.register_hparams
def lstm_base_batch_8k_hidden_2k():
    """Set of hyperparameters."""
    hparams = lstm_base_batch_8k_hidden_1k()

    hparams.hidden_size = 2048
    hparams.attention_layer_size = 2048

    return hparams


@registry.register_hparams
def lstm_base_batch_32k_hidden_512():
    """Set of hyperparameters."""
    hparams = lstm.lstm_attention()

    hparams.batch_size = 32768
    hparams.hidden_size = 512
    hparams.attention_layer_size = 512

    hparams.weight_decay = 1e-06
    hparams.dropout = 0.3

    hparams.optimizer = "AdamW"
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.999
    hparams.optimizer_adam_epsilon = 1e-8

    hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
    hparams.learning_rate_constant = 5e-5
    hparams.learning_rate_warmup_steps = 8000
    hparams.learning_rate_decay_steps = 80000

    # Attention
    hparams.add_hparam("attn_dim", 256)
    hparams.add_hparam("num_glimps", 2)

    # MLP
    hparams.add_hparam("num_mlp_layers", 1)
    hparams.add_hparam("mlp_dim", 512)

    # self attention parts
    hparams.norm_type = "layer"
    hparams.layer_preprocess_sequence = "n"
    hparams.layer_postprocess_sequence = "da"
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.1
    hparams.relu_dropout = 0.1
    hparams.image_hidden_size = 512
    hparams.add_hparam("num_encoder_layers", 1)
    # Attention-related flags.
    hparams.num_heads = 2
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("image_filter_size", 256)
    hparams.add_hparam("self_attention_type", "dot_product")
    hparams.add_hparam("scale_dotproduct", True)

    return hparams


@registry.register_hparams
def lstm_tall_ls0():
    """No label smoothing."""
    hparams = lstm.lstm_attention()

    hparams.batch_size = 16384
    hparams.hidden_size = 512
    hparams.attention_layer_size = 512

    hparams.num_hidden_layers = 4

    hparams.label_smoothing = 0.0

    hparams.optimizer = "AdamW"
    hparams.optimizer_adam_beta1 = 0.9
    hparams.optimizer_adam_beta2 = 0.999
    hparams.optimizer_adam_epsilon = 1e-8

    hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
    hparams.learning_rate_constant = 2e-4
    hparams.learning_rate_warmup_steps = 8000
    hparams.learning_rate_decay_steps = 80000

    # Attention
    hparams.add_hparam("attn_dim", 256)
    hparams.add_hparam("num_glimps", 2)

    # MLP
    hparams.add_hparam("num_mlp_layers", 1)
    hparams.add_hparam("mlp_dim", 512)

    # self attention parts
    hparams.norm_type = "layer"
    hparams.layer_preprocess_sequence = "n"
    hparams.layer_postprocess_sequence = "da"
    hparams.layer_prepostprocess_dropout = 0.3
    hparams.attention_dropout = 0.1
    hparams.relu_dropout = 0.1
    hparams.image_hidden_size = 2048
    hparams.add_hparam("num_encoder_layers", 1)
    # Attention-related flags.
    hparams.add_hparam("num_heads", 8)
    hparams.add_hparam("attention_key_channels", 0)
    hparams.add_hparam("attention_value_channels", 0)
    hparams.add_hparam("image_filter_size", 1024)
    hparams.add_hparam("self_attention_type", "dot_product")
    hparams.add_hparam("scale_dotproduct", True)

    return hparams
