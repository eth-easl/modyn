#include "../../../../../modyn/storage/internal/file_wrapper/binary_file_wrapper/binary_file_wrapper.h"
#include "gtest/gtest.h"

TEST(BinaryFileWrapperTest, get_label_native)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_label_native
    int label_native = get_label_native("test_file.bin", 0, sizeof(label), sizeof(label));
    ASSERT_EQ(label_native, 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

TEST(BinaryFileWrapperTest, get_label)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_label
    std::vector<unsigned char> data = get_data_from_file("test_file.bin");
    int label_native = get_label(data.data(), 0, sizeof(label), sizeof(label));
    ASSERT_EQ(label_native, 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

TEST(BinaryFileWrapperTest, get_all_labels_native)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_all_labels_native
    IntVector *labels = get_all_labels_native("test_file.bin", 3, sizeof(label), sizeof(label));
    ASSERT_EQ(labels->size, 3);
    ASSERT_EQ(labels->data[0], 5);
    ASSERT_EQ(labels->data[1], 5);
    ASSERT_EQ(labels->data[2], 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

TEST(BinaryFileWrapperTest, get_all_labels)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_all_labels
    std::vector<unsigned char> data = get_data_from_file("test_file.bin");
    IntVector *labels = get_all_labels(data.data(), 3, sizeof(label), sizeof(label));
    ASSERT_EQ(labels->size, 3);
    ASSERT_EQ(labels->data[0], 5);
    ASSERT_EQ(labels->data[1], 5);
    ASSERT_EQ(labels->data[2], 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

TEST(BinaryFileWrapperTest, get_samples_from_indices_native)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_samples_from_indices_native
    IntVector *indices = new IntVector;
    indices->size = 3;
    indices->data = new int[3];
    indices->data[0] = 0;
    indices->data[1] = 1;
    indices->data[2] = 2;
    CharVector *samples = get_samples_from_indices_native("test_file.bin", indices, sizeof(label), sizeof(label));
    ASSERT_EQ(samples->size, 3 * sizeof(label));
    ASSERT_EQ(samples->data[0], 5);
    ASSERT_EQ(samples->data[1], 5);
    ASSERT_EQ(samples->data[2], 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

TEST(BinaryFileWrapperTest, get_samples_from_indices)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_samples_from_indices
    std::vector<unsigned char> data = get_data_from_file("test_file.bin");
    IntVector *indices = new IntVector;
    indices->size = 3;
    indices->data = new int[3];
    indices->data[0] = 0;
    indices->data[1] = 1;
    indices->data[2] = 2;
    CharVector *samples = get_samples_from_indices(data.data(), indices, sizeof(label), sizeof(label));
    ASSERT_EQ(samples->size, 3 * sizeof(label));
    ASSERT_EQ(samples->data[0], 5);
    ASSERT_EQ(samples->data[1], 5);
    ASSERT_EQ(samples->data[2], 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

TEST(BinaryFileWrapperTest, int_from_bytes)
{
    // Test int_from_bytes
    unsigned char bytes[4] = {0, 0, 0, 5};
    int value = int_from_bytes(bytes, 4);
    ASSERT_EQ(value, 5);
}

TEST(BinaryFileWrapperTest, validate_request_indices)
{
    // Test validate_request_indices
    IntVector *indices = new IntVector;
    indices->size = 3;
    indices->data = new int[3];
    indices->data[0] = 0;
    indices->data[1] = 1;
    indices->data[2] = 2;
    bool result = validate_request_indices(3, indices);
    ASSERT_EQ(result, false);
    bool result2 = validate_request_indices(2, indices);
    ASSERT_EQ(result2, true);
}

TEST(BinaryFileWrapperTest, get_data_from_file)
{
    // Create a test file
    std::ofstream test_file;
    test_file.open("test_file.bin", std::ios::binary);
    int label = 5;
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));
    test_file.write(reinterpret_cast<char *>(&label), sizeof(label));

    // Test get_data_from_file
    std::vector<unsigned char> data = get_data_from_file("test_file.bin");
    ASSERT_EQ(data.size(), 3 * sizeof(label));
    ASSERT_EQ(data[0], 5);
    ASSERT_EQ(data[1], 5);
    ASSERT_EQ(data[2], 5);

    // Remove test file
    test_file.close();
    remove("test_file.bin");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}