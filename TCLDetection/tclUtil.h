//
//  tclUtil.h
//  TCLDetection
//
//  Created by Joseph McGrath on 9/6/18.

#ifndef tclUtil_h
#define tclUtil_h


// Contains general utilities for csv handling and string handling

// https://stackoverflow.com/questions/1120140/how-can-i-read-and-parse-csv-files-in-c?rq=1
class CSVRow
{
public:
    std::string const& operator[](std::size_t index) const
    {
        return m_data[index]; // returns value in index column of current row
    }
    std::size_t size() const
    {
        return m_data.size(); // will give the number of columns
    }
    void readNextRow(std::istream& str)
    {
        std::string         line;
        std::getline(str, line);
        
        std::stringstream   lineStream(line);
        std::string         cell;
        
        m_data.clear();
        while(std::getline(lineStream, cell, ','))
        {
            // Add cell to row vector
            m_data.push_back(cell);
        }
        // This checks for a trailing comma with no data after it.
        if (!lineStream && cell.empty())
        {
            // If there was a trailing comma then add an empty element.
            m_data.push_back("");
        }
    }
private:
    std::vector<std::string>    m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data)
{
    data.readNextRow(str);
    return str;
}

class CSVIterator
{
public:
    typedef std::input_iterator_tag     iterator_category;
    typedef CSVRow                      value_type;
    typedef std::size_t                 difference_type;
    typedef CSVRow*                     pointer;
    typedef CSVRow&                     reference;
    
    CSVIterator(std::istream& str)  :m_str(str.good()?&str:NULL) { ++(*this); }
    CSVIterator()                   :m_str(NULL) {}
    
    // Pre Increment
    CSVIterator& operator++()               {if (m_str) { if (!((*m_str) >> m_row)){m_str = NULL;}}return *this;}
    // Post increment
    CSVIterator operator++(int)             {CSVIterator    tmp(*this);++(*this);return tmp;}
    CSVRow const& operator*()   const       {return m_row;}
    CSVRow const* operator->()  const       {return &m_row;}
    
    bool operator==(CSVIterator const& rhs) {return ((this == &rhs) || ((this->m_str == NULL) && (rhs.m_str == NULL)));}
    bool operator!=(CSVIterator const& rhs) {return !((*this) == rhs);}
private:
    std::istream*       m_str;
    CSVRow              m_row;
};

// String utilities based on those in OSIRIS
class tclStringUtil {
public:
    /** Convert a string into any basic type.
     * @param rString A string
     * @return Template T
     */
    template < typename T > T fromString ( const std::string & rString ) ;
    
    /** Remove leading and trailing spaces and/or tabs.
     * @param rString A string
     * @return A string
     */
    std::string trim ( const std::string & rString )
    {
        std::string s = rString.substr(0,rString.find_last_not_of("\r\n")+1) ;
        size_t first = s.find_first_not_of(" \t") ;
        if ( first != std::string::npos )
            return s.substr(first,s.find_last_not_of(" \t") - first + 1) ;
        else
            return "" ;
    }
    
    /** Convert to lowercase.
     * @param rString A string
     * @return A string
     */
    std::string toLower ( const std::string & rString )
    {
        std::string out = rString ;
        std::transform(out.begin(),out.end(),out.begin(),::tolower) ;
        return out ;
    }
    
    
};

// Definition of fromString
template < typename T >
T tclStringUtil::fromString ( const std::string & rString )
{
    std::istringstream iss(rString) ;
    T out ;
    if ( ! ( iss >> out ) )
        throw std::runtime_error("Cannot convert " + rString + " into basic type") ;
    return out ;
}

// Specialization of function fromString() for boolean
template < >
inline bool tclStringUtil::fromString<bool> ( const std::string & rString )
{
    std::string s = trim(toLower(rString)) ;
    if ( s == "yes" || s == "true" || s == "on" || s == "y" || s == "1" )
        return true ;
    else if ( s == "no" || s == "false" || s == "off" || s == "n" || s == "0" )
        return false ;
    else
        throw std::runtime_error("Cannot convert " + rString + " into boolean") ;
}



#endif /* tclUtil_h */
