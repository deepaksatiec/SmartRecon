from numpy.testing import assert_array_equal
from helper.encoding import MultiColumnLabelEncoder, OneHotEncoder


def test_label_encoding(data, feat_types):
    categorical_feat = [key for key, value in feat_types.items() if value.lower() == 'categorical']
    enc = MultiColumnLabelEncoder(columns=categorical_feat)

    global transformed_data
    transformed_data = enc.fit_transform(data)
    assert not transformed_data.empty

    enc.fit(data)
    assert_array_equal(transformed_data, enc.transform(data))

    assert transformed_data.shape[0] == data.shape[0]


def test_one_hot_encoding(data, feat_types):
    enc = OneHotEncoder(minimum_fraction=0.01,
                        categorical_features=[(feat_types[value] == 'categorical') for value in feat_types],
                        sparse=False)
    encoded_data = enc.fit_transform(transformed_data)

    assert encoded_data.size != 0
    assert encoded_data.shape[0] == data.shape[0]

    enc.fit(transformed_data)
    assert_array_equal(encoded_data, enc.transform(transformed_data))
