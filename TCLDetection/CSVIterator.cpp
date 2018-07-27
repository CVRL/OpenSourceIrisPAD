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
    
    bool lookingForEndquote = false;
    int startQuoteIdx = -1;
    int endQuoteIdx = -1;
    
    m_data.clear();
    while(std::getline(lineStream, cell, ','))
    {
        // Add cell to row vector
        m_data.push_back(cell);
        
        // Look for presence of quotes within a cell, would indicate a list of items (assume that there will not be two sets in the same cell)
        if (!(cell.find("\"") == std::string::npos)) {
            if (lookingForEndquote) {
                endQuoteIdx = (int)m_data.size() - 1;
                
                // Append strings within set of quotes and erase separate vector cells
                for (int i = 1; i < ((endQuoteIdx - startQuoteIdx) + 1); i++) {
                    // Add cell to the immediate right of start, then remove that cell so next cell is always at (start + 1)
                    m_data.at(startQuoteIdx) = m_data.at(startQuoteIdx).append("," + m_data.at(startQuoteIdx + 1));
                    m_data.erase(m_data.begin() + startQuoteIdx + 1);
                }
                // Remove extra commas
                size_t commaIdx;
                while ((commaIdx = m_data[startQuoteIdx].find(",")) != std::string::npos) {
                    m_data[startQuoteIdx].erase(commaIdx,1);
                }
                // Remove quotation marks
                size_t quoteIdx;
                while ((quoteIdx = m_data[startQuoteIdx].find("\"")) != std::string::npos) {
                    m_data[startQuoteIdx].erase(quoteIdx,1);
                }
                // Reset control
                lookingForEndquote = false;

            } else {
                startQuoteIdx = (int)m_data.size() - 1;
                lookingForEndquote = true;
            }
        }
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

