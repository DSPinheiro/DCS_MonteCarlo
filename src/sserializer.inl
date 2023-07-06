// cgodinho 15/05/2023

#include <vector>
#include <string>
#include <iterator>
#include <iostream>

// No C++20, so no concepts (in reality S can be any pod type)
template<typename T>
inline void serialize_data(std::vector<uint8_t>& buffer, const T& obj)
{
    // Copy directly to buffer
    if constexpr(std::is_pod<T>::value)
    {
        buffer.reserve(buffer.capacity() + sizeof(T));
        const uint8_t* data = reinterpret_cast<const uint8_t*>(&obj);
        std::copy(data, data + sizeof(T), std::back_inserter(buffer));
    }
    else
    {
        static_assert(!std::is_same<T,T>::value, "Cannot directly serialize non-POD data.");
    }
}

// Specialize std::string for serialization
template<>
inline void serialize_data(std::vector<uint8_t>& buffer, const std::string& obj)
{
    serialize_data(buffer, obj.size());
    buffer.reserve(buffer.capacity() + obj.size());
    std::copy(obj.begin(), obj.end(), std::back_inserter(buffer));
}

template<typename T>
inline void consume_buffer(std::vector<uint8_t>& buffer, T& dest)
{
    // Copy directly from buffer
    if constexpr(std::is_pod<T>::value)
    {
        if(buffer.empty())
        {
            std::cout << "Buffer is empty. No data was consumed." << std::endl;
            return;
        }

        uint8_t* data = reinterpret_cast<uint8_t*>(&dest);
        std::copy(buffer.begin(), buffer.begin() + sizeof(T), data);
        buffer.erase(buffer.begin(), buffer.begin() + sizeof(T));
    }
    else
    {
        static_assert(!std::is_same<T,T>::value, "Cannot directly read buffer into non-POD data.");
    }
}

template<>
inline void consume_buffer(std::vector<uint8_t>& buffer, std::string& dest)
{
    std::string::size_type string_size;
    consume_buffer(buffer, string_size);

    // Read string_size bytes
    dest.clear();
    dest.reserve(string_size);
    std::copy(buffer.begin(), buffer.begin() + string_size, std::back_inserter(dest));
    buffer.erase(buffer.begin(), buffer.begin() + string_size);
}
