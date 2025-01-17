/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "GIDI.hpp"
#include <HAPI.hpp>

namespace GIDI {

/*! \class Suite
 * This class is used to store a list (i.e., suite) of similar type **GNDS** nodes.
*/

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Suite::Suite( ) :
        Ancestry( "" ) {

}

/* *********************************************************************************************************//**
 * @param a_moniker             [in]    The **GNDS** moniker for the Suite instance.
 ***********************************************************************************************************/

Suite::Suite( std::string const &a_moniker ) :
        Ancestry( a_moniker ) {

}

/* *********************************************************************************************************//**
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_moniker             [in]    The **GNDS** moniker for the Suite instance.
 * @param a_node                [in]    The HAPI::Node to be parsed and used to construct the Product.
 * @param a_setupInfo           [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops                [in]    The *external* PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs        [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                      This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parseSuite          [in]    This function to call to parse each sub-node.
 * @param a_styles              [in]    The <**styles**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

Suite::Suite( Construction::Settings const &a_construction, std::string const &a_moniker, HAPI::Node const &a_node, SetupInfo &a_setupInfo,
		PoPI::Database const &a_pops, PoPI::Database const &a_internalPoPs, parseSuite a_parseSuite, Styles::Suite const *a_styles ) :
        Ancestry( a_moniker ),
        m_styles( a_styles ) {

    HAPI::Node const node = a_node.child( a_moniker.c_str( ) );

    if( ! node.empty( ) ) parse( a_construction, node, a_setupInfo, a_pops, a_internalPoPs, a_parseSuite, a_styles );
}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Suite::~Suite( ) {

    for( std::vector<Form *>::const_iterator iter = m_forms.begin( ); iter < m_forms.end( ); ++iter ) delete *iter;
}

/* *********************************************************************************************************//**
 *
 * @param a_construction        [in]    Used to pass user options to the constructor.
 * @param a_node                [in]    The HAPI::Node to be parsed and used to construct the Product.
 * @param a_setupInfo           [in]    Information create my the Protare constructor to help in parsing.
 * @param a_pops                [in]    The *external* PoPI::Database instance used to get particle indices and possibly other particle information.
 * @param a_internalPoPs        [in]    The *internal* PoPI::Database instance used to get particle indices and possibly other particle information.
 *                                      This is the <**PoPs**> node under the <**reactionSuite**> node.
 * @param a_parseSuite          [in]    This function to call to parse each sub-node.
 * @param a_styles              [in]    The <**styles**> node under the <**reactionSuite**> node.
 ***********************************************************************************************************/

void Suite::parse( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, PoPI::Database const &a_pops,
		PoPI::Database const &a_internalPoPs, parseSuite a_parseSuite, GIDI::Styles::Suite const *a_styles ) {

    for( HAPI::Node child = a_node.first_child( ); !child.empty( ); child.to_next_sibling( ) ) {
        std::string name( child.name( ) );

        Form *form = a_parseSuite( a_construction, this, child, a_setupInfo, a_pops, a_internalPoPs, name, a_styles );
        if( form != nullptr ) add( form );
    }
}

/* *********************************************************************************************************//**
 * Returns the index of the of the node in *this* that has label *a_label*.

 * @return                      [in]    The index of the node with label *a_label* in *this*.
 ***********************************************************************************************************/

int Suite::operator[]( std::string const &a_label ) const {

    std::map<std::string, int>::const_iterator iter = m_map.find( a_label );
    if( iter == m_map.end( ) ) {
        throw Exception( "form '" + a_label + "' not in database." );
    }

    return( iter->second );
}

/* *********************************************************************************************************//**
 * Adds the node *a_form* to *this*.
 *
 * @param a_form                [in]    The form to add.
 ***********************************************************************************************************/

void Suite::add( Form *a_form ) {

    int i1 = 0;

    for( Suite::iterator iter = begin( ); iter != end( ); ++iter, ++i1 ) {
        if( (*iter)->label( ) == a_form->label( ) ) {
            m_forms[i1] = a_form;
            return;
        }
    }
    m_map[a_form->label( )] = (int) m_forms.size( );
    m_forms.push_back( a_form );
    a_form->setAncestor( this );
}

/* *********************************************************************************************************//**
 * Returns the iterator to the node with label *a_label*.
 *
 * @param a_label               [in]    The label of the node to find.
 *
 * @return                              The iterator to the node with label *a_label*.
 ***********************************************************************************************************/

Suite::iterator Suite::find( std::string const &a_label ) {

    for( Suite::iterator iter = begin( ); iter != end( ); ++iter ) {
        if( (*iter)->label( ) == a_label ) return( iter );
    }
    return( end( ) );
}

/* *********************************************************************************************************//**
 * Returns the iterator to the node with label *a_label*.
 *
 * @param a_label               [in]    The label of the node to find.
 *
 * @return                              The iterator to the node with label *a_label*.
 ***********************************************************************************************************/

Suite::const_iterator Suite::find( std::string const &a_label ) const {

    for( Suite::const_iterator iter = begin( ); iter != end( ); ++iter ) {
        if( (*iter)->label( ) == a_label ) return( iter );
    }
    return( end( ) );
}

/* *********************************************************************************************************//**
 * Returns a list of iterators to the nodes in *this* that have **GNDS** moniker *a_moniker*.
 *
 * @param a_moniker             [in]    The moniker to search for.
 *
 * @return                              List of iterators to the nodes in *this* that have moniker *a_moniker*.
 ***********************************************************************************************************/

std::vector<Suite::iterator> Suite::findAllOfMoniker( std::string const &a_moniker ) {

    std::vector<Suite::iterator> iters;

    for( Suite::iterator iter = m_forms.begin( ); iter != m_forms.end( ); ++iter ) {
        if( (*iter)->moniker( ) == a_moniker ) iters.push_back( iter );
    }

    return( iters );
}

/* *********************************************************************************************************//**
 * Returns a list of iterators to the nodes in *this* that have **GNDS** moniker *a_moniker*.
 *
 * @param a_moniker             [in]    The moniker to search for.
 *
 * @return                              List of iterators to the nodes in *this* that have moniker *a_moniker*.
 ***********************************************************************************************************/

std::vector<Suite::const_iterator> Suite::findAllOfMoniker( std::string const &a_moniker ) const {

    std::vector<Suite::const_iterator> iters;

    for( Suite::const_iterator iter = m_forms.begin( ); iter != m_forms.end( ); ++iter ) {
        if( (*iter)->moniker( ) == a_moniker ) iters.push_back( iter );
    }

    return( iters );
}

/* *********************************************************************************************************//**
 * Only for internal use. Called by ProtareTNSL instance to zero the lower energy multi-group data covered by the ProtareSingle that
 * contains the TNSL data covers the lower energy multi-group data.
 *
 * @param a_maximumTNSL_MultiGroupIndex     [in]    A map that contains labels for heated multi-group data and the last valid group boundary
 *                                                  for the TNSL data for that boundary.
 ***********************************************************************************************************/

void Suite::modifiedMultiGroupElasticForTNSL( std::map<std::string,std::size_t> a_maximumTNSL_MultiGroupIndex ) {

    for( auto iter = a_maximumTNSL_MultiGroupIndex.begin( ); iter != a_maximumTNSL_MultiGroupIndex.end( ); ++iter ) {
        auto formIter = find( iter->first );

        if( formIter == end( ) ) continue;

        if( (*formIter)->type( ) == FormType::gridded1d ) {
            reinterpret_cast<Functions::Gridded1d *>( (*formIter) )->modifiedMultiGroupElasticForTNSL( iter->second ); }
        else if( (*formIter)->type( ) == FormType::gridded3d ) {
            reinterpret_cast<Functions::Gridded3d *>( (*formIter) )->modifiedMultiGroupElasticForTNSL( iter->second );
        }
    }
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or nullptr if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or nullptr if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry *Suite::findInAncestry3( std::string const &a_item ) {

    std::size_t index( a_item.find( '=' ) ), lastQuote = a_item.size( ) - 2;

    if( index == std::string::npos ) return( nullptr );
    ++index;
    if( index > lastQuote ) throw Exception( "Suite::findInAncestry3: invalide xlink" );
    if( a_item[index] != '\'' ) throw Exception( "Suite::findInAncestry3: invalid xlink, missing '." );
    ++index;
    if( a_item[lastQuote]  != '\'' ) throw Exception( "Suite::findInAncestry3: invalid xlink, missing endl '." );

    std::string label( a_item.substr( index, lastQuote - index ) );

    return( get<Ancestry>( label ) );
}

/* *********************************************************************************************************//**
 * Used by Ancestry to tranverse GNDS nodes. This method returns a pointer to a derived class' a_item member or nullptr if none exists.
 *
 * @param a_item    [in]    The name of the class member whose pointer is to be return.
 * @return                  The pointer to the class member or nullptr if class does not have a member named a_item.
 ***********************************************************************************************************/

Ancestry const *Suite::findInAncestry3( std::string const &a_item ) const {

    std::size_t index( a_item.find( '=' ) ), lastQuote = a_item.size( ) - 2;

    if( index == std::string::npos ) return( nullptr );
    ++index;
    if( index > lastQuote ) throw Exception( "Suite::findInAncestry3: invalide xlink" );
    if( a_item[index] != '\'' ) throw Exception( "Suite::findInAncestry3: invalid xlink, missing '." );
    ++index;
    if( a_item[lastQuote]  != '\'' ) throw Exception( "Suite::findInAncestry3: invalid xlink, missing endl '." );

    std::string label( a_item.substr( index, lastQuote - index ) );

    return( get<Ancestry>( label ) );
}

/* *********************************************************************************************************//**
 * Fills the argument *a_writeInfo* with the XML lines that represent *this*. Recursively enters each sub-node.
 *
 * @param       a_writeInfo         [in/out]    Instance containing incremental indentation and other information and stores the appended lines.
 * @param       a_indent            [in]        The amount to indent *this* node.
 ***********************************************************************************************************/

void Suite::toXMLList( WriteInfo &a_writeInfo, std::string const &a_indent ) const {

    std::string indent2 = a_writeInfo.incrementalIndent( a_indent );

    if( size( ) == 0 ) return;

    std::string XMLLine( a_indent + "<" + moniker( ) + ">" );
    a_writeInfo.push_back( XMLLine );

    for( Suite::const_iterator iter = m_forms.begin( ); iter != m_forms.end( ); ++iter ) (*iter)->toXMLList( a_writeInfo, indent2 );

    a_writeInfo.addNodeEnder( moniker( ) );
}
 
/* *********************************************************************************************************//**
 * Prints the list of node labels to std::cout.
 *
 * @param a_header              [in]    A string printed before the list of labels is printed.
 ***********************************************************************************************************/

void Suite::printFormLabels( std::string const &a_header ) const {

    std::cout << a_header << ": size = " << size( ) << std::endl;
    
    for( Suite::const_iterator iter = m_forms.begin( ); iter != m_forms.end( ); ++iter ) 
            std::cout << "    " << (*iter)->label( ) << std::endl;
}

}
