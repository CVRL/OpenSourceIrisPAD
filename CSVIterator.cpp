//https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c?rq=1
// CSVRow and CSVIterator taken from this stackoverflow question

#include "CSVIterator.hpp"


std::string const& CSVRow::operator[](std::size_t index) const
{
    return m_data[index]; //Returns value in index column of current row
}
std::size_t CSVRow::size() const
{
    return m_data.size(); //will give the number of columns
}
void CSVRow::readNextRow(std::istream& str)
{
    std::string         line;
    std::getline(str, line);
    
    std::stringstream   lineStream(line);
    std::string         cell;
    
    m_data.clear();
    while(std::getline(lineStream, cell, ','))
    {
        m_data.push_back(cell);
    }
    // This checks for a trailing comma with no data after it.
    if (!lineStream && cell.empty())
    {
        // If there was a trailing comma then add an empty element.
        m_data.push_back("");
    }
}

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}


CSVIterator::CSVIterator(std::istream& str)  :m_str(str.good()?&str:NULL) { ++(*this); }
CSVIterator::CSVIterator()                   :m_str(NULL) {}
    
// Pre Increment
CSVIterator& CSVIterator::operator++()               {if (m_str) { if (!((*m_str) >> m_row)){m_str = NULL;}}return *this;}
// Post increment
CSVIterator CSVIterator::operator++(int)             {CSVIterator    tmp(*this);++(*this);return tmp;}
CSVRow const& CSVIterator::operator*()   const       {return m_row;}
CSVRow const* CSVIterator::operator->()  const       {return &m_row;}
    
bool CSVIterator::operator==(CSVIterator const& rhs) {return ((this == &rhs) || ((this->m_str == NULL) && (rhs.m_str == NULL)));}
bool CSVIterator::operator!=(CSVIterator const& rhs) {return !((*this) == rhs);}

