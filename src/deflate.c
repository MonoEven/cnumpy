/**
 * Raw RFC 1951 DEFLATE decoder used by NPZ ZIP members.
 *
 * ZIP method 8 stores the DEFLATE bitstream without a zlib or gzip wrapper.
 * This decoder deliberately has a narrow interface: the ZIP directory supplies
 * the exact output size, and decoding succeeds only when the stream produces
 * exactly that many bytes.
 */
#include "../include/cnumpy/cnumpy_internal.h"
#include <string.h>

#define DEFLATE_MAX_BITS 15
#define DEFLATE_MAX_LITLEN_SYMBOLS 288
#define DEFLATE_MAX_DISTANCE_SYMBOLS 32
#define DEFLATE_MAX_CODELEN_SYMBOLS 19
#define DEFLATE_MAX_TREE_NODES(symbols) (1 + (symbols) * DEFLATE_MAX_BITS)

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t offset;
    uint8_t bits;
    int bit_count;
} DeflateBitReader;

typedef struct {
    int16_t child[2];
    int16_t symbol;
} DeflateHuffmanNode;

typedef struct {
    DeflateHuffmanNode *nodes;
    int capacity;
    int count;
} DeflateHuffmanTree;

static bool deflate_error(const char *function_name, const char *detail) {
    cnp_set_error(
        CNP_ERR_IO, function_name, "Invalid DEFLATE stream: %s", detail);
    return false;
}

static bool deflate_read_bits(
    DeflateBitReader *reader, int count, uint32_t *value) {
    uint32_t result = 0;
    for (int bit = 0; bit < count; ++bit) {
        if (reader->bit_count == 0) {
            if (reader->offset == reader->size) return false;
            reader->bits = reader->data[reader->offset++];
            reader->bit_count = 8;
        }
        result |= (uint32_t)(reader->bits & 1u) << bit;
        reader->bits >>= 1;
        --reader->bit_count;
    }
    *value = result;
    return true;
}

static void deflate_align_to_byte(DeflateBitReader *reader) {
    reader->bits = 0;
    reader->bit_count = 0;
}

static int deflate_new_node(DeflateHuffmanTree *tree) {
    if (tree->count == tree->capacity) return -1;
    int index = tree->count++;
    tree->nodes[index].child[0] = -1;
    tree->nodes[index].child[1] = -1;
    tree->nodes[index].symbol = -1;
    return index;
}

static bool deflate_build_tree(
    DeflateHuffmanTree *tree, const uint8_t *lengths, int symbol_count,
    const char *function_name) {
    uint16_t counts[DEFLATE_MAX_BITS + 1] = {0};
    uint16_t next_code[DEFLATE_MAX_BITS + 1] = {0};
    int populated = 0;
    tree->count = 0;
    if (deflate_new_node(tree) < 0)
        return deflate_error(function_name, "Huffman tree capacity exceeded");

    for (int symbol = 0; symbol < symbol_count; ++symbol) {
        uint8_t length = lengths[symbol];
        if (length > DEFLATE_MAX_BITS)
            return deflate_error(function_name, "invalid Huffman code length");
        if (length != 0) {
            ++counts[length];
            ++populated;
        }
    }
    if (populated == 0)
        return deflate_error(function_name, "empty Huffman alphabet");

    int available = 1;
    for (int bits = 1; bits <= DEFLATE_MAX_BITS; ++bits) {
        available = available * 2 - counts[bits];
        if (available < 0)
            return deflate_error(function_name, "oversubscribed Huffman tree");
    }

    uint16_t code = 0;
    for (int bits = 1; bits <= DEFLATE_MAX_BITS; ++bits) {
        code = (uint16_t)((code + counts[bits - 1]) << 1);
        next_code[bits] = code;
    }

    for (int symbol = 0; symbol < symbol_count; ++symbol) {
        int length = lengths[symbol];
        if (length == 0) continue;
        uint16_t symbol_code = next_code[length]++;
        int node = 0;
        for (int position = length - 1; position >= 0; --position) {
            if (tree->nodes[node].symbol >= 0)
                return deflate_error(
                    function_name, "Huffman code extends a leaf");
            int bit = (symbol_code >> position) & 1;
            int child = tree->nodes[node].child[bit];
            if (child < 0) {
                child = deflate_new_node(tree);
                if (child < 0)
                    return deflate_error(
                        function_name, "Huffman tree capacity exceeded");
                tree->nodes[node].child[bit] = (int16_t)child;
            }
            node = child;
        }
        if (tree->nodes[node].symbol >= 0 ||
                tree->nodes[node].child[0] >= 0 ||
                tree->nodes[node].child[1] >= 0)
            return deflate_error(function_name, "duplicate Huffman code");
        tree->nodes[node].symbol = (int16_t)symbol;
    }
    return true;
}

static bool deflate_decode_symbol(
    DeflateBitReader *reader, const DeflateHuffmanTree *tree,
    int *symbol, const char *function_name) {
    int node = 0;
    for (int depth = 0; depth < DEFLATE_MAX_BITS; ++depth) {
        uint32_t bit;
        if (!deflate_read_bits(reader, 1, &bit))
            return deflate_error(function_name, "truncated Huffman code");
        node = tree->nodes[node].child[bit];
        if (node < 0)
            return deflate_error(function_name, "unknown Huffman code");
        if (tree->nodes[node].symbol >= 0) {
            *symbol = tree->nodes[node].symbol;
            return true;
        }
    }
    return deflate_error(function_name, "Huffman code is too long");
}

static bool deflate_build_fixed_trees(
    DeflateHuffmanTree *literal_tree, DeflateHuffmanTree *distance_tree,
    const char *function_name) {
    uint8_t literal_lengths[DEFLATE_MAX_LITLEN_SYMBOLS];
    uint8_t distance_lengths[DEFLATE_MAX_DISTANCE_SYMBOLS];
    for (int symbol = 0; symbol <= 143; ++symbol)
        literal_lengths[symbol] = 8;
    for (int symbol = 144; symbol <= 255; ++symbol)
        literal_lengths[symbol] = 9;
    for (int symbol = 256; symbol <= 279; ++symbol)
        literal_lengths[symbol] = 7;
    for (int symbol = 280; symbol < DEFLATE_MAX_LITLEN_SYMBOLS; ++symbol)
        literal_lengths[symbol] = 8;
    memset(distance_lengths, 5, sizeof(distance_lengths));
    return deflate_build_tree(
            literal_tree, literal_lengths,
            DEFLATE_MAX_LITLEN_SYMBOLS, function_name) &&
        deflate_build_tree(
            distance_tree, distance_lengths,
            DEFLATE_MAX_DISTANCE_SYMBOLS, function_name);
}

static bool deflate_build_dynamic_trees(
    DeflateBitReader *reader,
    DeflateHuffmanTree *literal_tree, DeflateHuffmanTree *distance_tree,
    const char *function_name) {
    static const uint8_t code_length_order[DEFLATE_MAX_CODELEN_SYMBOLS] = {
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5,
        11, 4, 12, 3, 13, 2, 14, 1, 15
    };
    uint32_t hlit_bits;
    uint32_t hdist_bits;
    uint32_t hclen_bits;
    if (!deflate_read_bits(reader, 5, &hlit_bits) ||
            !deflate_read_bits(reader, 5, &hdist_bits) ||
            !deflate_read_bits(reader, 4, &hclen_bits))
        return deflate_error(function_name, "truncated dynamic header");
    int literal_count = (int)hlit_bits + 257;
    int distance_count = (int)hdist_bits + 1;
    int code_length_count = (int)hclen_bits + 4;
    if (literal_count > 286)
        return deflate_error(function_name, "invalid literal alphabet size");

    uint8_t code_lengths[DEFLATE_MAX_CODELEN_SYMBOLS] = {0};
    for (int index = 0; index < code_length_count; ++index) {
        uint32_t length;
        if (!deflate_read_bits(reader, 3, &length))
            return deflate_error(
                function_name, "truncated code-length alphabet");
        code_lengths[code_length_order[index]] = (uint8_t)length;
    }

    DeflateHuffmanNode code_length_nodes[
        DEFLATE_MAX_TREE_NODES(DEFLATE_MAX_CODELEN_SYMBOLS)];
    DeflateHuffmanTree code_length_tree = {
        code_length_nodes,
        DEFLATE_MAX_TREE_NODES(DEFLATE_MAX_CODELEN_SYMBOLS),
        0
    };
    if (!deflate_build_tree(
            &code_length_tree, code_lengths,
            DEFLATE_MAX_CODELEN_SYMBOLS, function_name))
        return false;

    uint8_t lengths[
        DEFLATE_MAX_LITLEN_SYMBOLS + DEFLATE_MAX_DISTANCE_SYMBOLS] = {0};
    int total = literal_count + distance_count;
    int index = 0;
    while (index < total) {
        int symbol;
        if (!deflate_decode_symbol(
                reader, &code_length_tree, &symbol, function_name))
            return false;
        if (symbol <= 15) {
            lengths[index++] = (uint8_t)symbol;
            continue;
        }

        uint32_t extra;
        int repeat;
        uint8_t repeated_length;
        if (symbol == 16) {
            if (index == 0 || !deflate_read_bits(reader, 2, &extra))
                return deflate_error(
                    function_name, "invalid previous-length repeat");
            repeat = 3 + (int)extra;
            repeated_length = lengths[index - 1];
        } else if (symbol == 17) {
            if (!deflate_read_bits(reader, 3, &extra))
                return deflate_error(
                    function_name, "truncated zero-length repeat");
            repeat = 3 + (int)extra;
            repeated_length = 0;
        } else if (symbol == 18) {
            if (!deflate_read_bits(reader, 7, &extra))
                return deflate_error(
                    function_name, "truncated long zero-length repeat");
            repeat = 11 + (int)extra;
            repeated_length = 0;
        } else {
            return deflate_error(function_name, "invalid repeat symbol");
        }
        if (repeat > total - index)
            return deflate_error(
                function_name, "code-length repeat exceeds alphabet");
        memset(lengths + index, repeated_length, (size_t)repeat);
        index += repeat;
    }

    if (lengths[256] == 0)
        return deflate_error(function_name, "missing end-of-block symbol");
    if (!deflate_build_tree(
            literal_tree, lengths, literal_count, function_name))
        return false;

    bool has_distance_symbol = false;
    for (int symbol = 0; symbol < distance_count; ++symbol) {
        if (lengths[literal_count + symbol] != 0) {
            has_distance_symbol = true;
            break;
        }
    }
    if (!has_distance_symbol) {
        /* A distance alphabet with one zero-length code is legal when unused. */
        distance_tree->count = 0;
        deflate_new_node(distance_tree);
        return true;
    }
    return deflate_build_tree(
        distance_tree, lengths + literal_count,
        distance_count, function_name);
}

static bool deflate_decode_compressed_block(
    DeflateBitReader *reader,
    const DeflateHuffmanTree *literal_tree,
    const DeflateHuffmanTree *distance_tree,
    uint8_t *destination, size_t destination_size, size_t *output_offset,
    const char *function_name) {
    static const uint16_t length_base[29] = {
        3, 4, 5, 6, 7, 8, 9, 10,
        11, 13, 15, 17, 19, 23, 27, 31,
        35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258
    };
    static const uint8_t length_extra[29] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 2, 2,
        3, 3, 3, 3, 4, 4, 4, 4,
        5, 5, 5, 5, 0
    };
    static const uint16_t distance_base[30] = {
        1, 2, 3, 4, 5, 7, 9, 13,
        17, 25, 33, 49, 65, 97, 129, 193,
        257, 385, 513, 769, 1025, 1537, 2049, 3073,
        4097, 6145, 8193, 12289, 16385, 24577
    };
    static const uint8_t distance_extra[30] = {
        0, 0, 0, 0, 1, 1, 2, 2,
        3, 3, 4, 4, 5, 5, 6, 6,
        7, 7, 8, 8, 9, 9, 10, 10,
        11, 11, 12, 12, 13, 13
    };

    for (;;) {
        int symbol;
        if (!deflate_decode_symbol(
                reader, literal_tree, &symbol, function_name))
            return false;
        if (symbol < 256) {
            if (*output_offset == destination_size)
                return deflate_error(
                    function_name, "literal exceeds declared output size");
            destination[(*output_offset)++] = (uint8_t)symbol;
            continue;
        }
        if (symbol == 256) return true;
        if (symbol < 257 || symbol > 285)
            return deflate_error(function_name, "invalid length symbol");

        int length_index = symbol - 257;
        uint32_t extra = 0;
        if (!deflate_read_bits(
                reader, length_extra[length_index], &extra))
            return deflate_error(function_name, "truncated match length");
        size_t length = length_base[length_index] + (size_t)extra;

        int distance_symbol;
        if (!deflate_decode_symbol(
                reader, distance_tree, &distance_symbol, function_name))
            return false;
        if (distance_symbol < 0 || distance_symbol >= 30)
            return deflate_error(function_name, "invalid distance symbol");
        extra = 0;
        if (!deflate_read_bits(
                reader, distance_extra[distance_symbol], &extra))
            return deflate_error(function_name, "truncated match distance");
        size_t distance = distance_base[distance_symbol] + (size_t)extra;
        if (distance > *output_offset)
            return deflate_error(
                function_name, "match distance precedes output");
        if (length > destination_size - *output_offset)
            return deflate_error(
                function_name, "match exceeds declared output size");
        for (size_t index = 0; index < length; ++index) {
            destination[*output_offset] =
                destination[*output_offset - distance];
            ++*output_offset;
        }
    }
}

bool cnp_inflate_raw(
    const uint8_t *source, size_t source_size,
    uint8_t *destination, size_t destination_size,
    size_t *written, const char *function_name) {
    if ((!source && source_size != 0) ||
            (!destination && destination_size != 0) || !written) {
        return deflate_error(function_name, "invalid decoder arguments");
    }
    *written = 0;
    DeflateBitReader reader = {source, source_size, 0, 0, 0};

    DeflateHuffmanNode literal_nodes[
        DEFLATE_MAX_TREE_NODES(DEFLATE_MAX_LITLEN_SYMBOLS)];
    DeflateHuffmanNode distance_nodes[
        DEFLATE_MAX_TREE_NODES(DEFLATE_MAX_DISTANCE_SYMBOLS)];
    DeflateHuffmanTree literal_tree = {
        literal_nodes,
        DEFLATE_MAX_TREE_NODES(DEFLATE_MAX_LITLEN_SYMBOLS),
        0
    };
    DeflateHuffmanTree distance_tree = {
        distance_nodes,
        DEFLATE_MAX_TREE_NODES(DEFLATE_MAX_DISTANCE_SYMBOLS),
        0
    };

    bool final_block = false;
    while (!final_block) {
        uint32_t final_bit;
        uint32_t block_type;
        if (!deflate_read_bits(&reader, 1, &final_bit) ||
                !deflate_read_bits(&reader, 2, &block_type))
            return deflate_error(function_name, "truncated block header");
        final_block = final_bit != 0;

        if (block_type == 0) {
            deflate_align_to_byte(&reader);
            if (reader.size - reader.offset < 4)
                return deflate_error(function_name, "truncated stored block");
            uint16_t length = (uint16_t)reader.data[reader.offset] |
                ((uint16_t)reader.data[reader.offset + 1] << 8);
            uint16_t complement = (uint16_t)reader.data[reader.offset + 2] |
                ((uint16_t)reader.data[reader.offset + 3] << 8);
            reader.offset += 4;
            if ((uint16_t)(length ^ UINT16_MAX) != complement)
                return deflate_error(
                    function_name, "stored block length mismatch");
            if ((size_t)length > reader.size - reader.offset)
                return deflate_error(
                    function_name, "truncated stored block data");
            if ((size_t)length > destination_size - *written)
                return deflate_error(
                    function_name, "stored block exceeds output size");
            memcpy(destination + *written, reader.data + reader.offset, length);
            reader.offset += length;
            *written += length;
            continue;
        }

        if (block_type == 1) {
            if (!deflate_build_fixed_trees(
                    &literal_tree, &distance_tree, function_name))
                return false;
        } else if (block_type == 2) {
            if (!deflate_build_dynamic_trees(
                    &reader, &literal_tree, &distance_tree, function_name))
                return false;
        } else {
            return deflate_error(function_name, "reserved block type");
        }
        if (!deflate_decode_compressed_block(
                &reader, &literal_tree, &distance_tree,
                destination, destination_size, written, function_name))
            return false;
    }

    if (*written != destination_size)
        return deflate_error(
            function_name, "decoded size differs from ZIP directory");
    return true;
}
