#ifndef CSVIterator_hpp
#define CSVIterator_hpp

#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

class CSVRow {
public:
    std::string const& operator[](std::size_t index) const;
    std::size_t size() const;
    void readNextRow(std::istream& str);
private:
    std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data);

class CSVIterator {
public:
    typedef std::input_iterator_tag     iterator_category;
    typedef CSVRow                      value_type;
    typedef std::size_t                 difference_type;
    typedef CSVRow*                     pointer;
    typedef CSVRow&                     reference;
    
    CSVIterator(std::istream& str);
    CSVIterator();
    
    // Pre Increment
    CSVIterator& operator++();
    // Post increment
    CSVIterator operator++(int);
    CSVRow const& operator*()   const;
    CSVRow const* operator->()  const;
    
    bool operator==(CSVIterator const& rhs);
    bool operator!=(CSVIterator const& rhs);
private:
    std::istream*       m_str;
    CSVRow              m_row;
};

#endif /* CSVIterator_hpp */
