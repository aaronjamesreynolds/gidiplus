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

namespace Functions {

/*! \class Branching1d
 * Class for the GNDS <**branching1d**> node.
 */

/* *********************************************************************************************************//**
 *
 * @param a_construction    [in]    Used to pass user options for parsing.
 * @param a_node            [in]    The **HAPI::Node** to be parsed and used to construct the XYs2d.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_parent          [in]    The parent GIDI::Suite.
 ***********************************************************************************************************/

Branching1d::Branching1d( Construction::Settings const &a_construction, HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent ) :
        Function1dForm( a_construction, a_node, a_setupInfo, FormType::branching1d, a_parent ),
        m_pids( a_node.child( GIDI_pidsChars ), a_setupInfo, nullptr ),
        m_multiplicity( 0.0 ) {

}

/* *********************************************************************************************************//**
 ***********************************************************************************************************/

Branching1d::~Branching1d( ) {

}

/* *********************************************************************************************************//**
 * Returns the domain minimum for the instance.
 *
 * @return          The domain minimum for the instance.
 ***********************************************************************************************************/

double Branching1d::domainMin( ) const {

    return( 0.0 );              // FIXME
}

/* *********************************************************************************************************//**
 * Returns the domain maximum for the instance.
 *
 * @return              The domain maximum for the instance.
 ***********************************************************************************************************/

double Branching1d::domainMax( ) const {

    return( 1.0 );              // FIXME
}

/* *********************************************************************************************************//**
 * Returns the value of the function *f(x1)* at the specified point and *a_x1*.
 * **This is currently not implemented**.
 *
 * @param a_x1              [in]    The value of the **x1** axis.
 * @return                          The value of the function evaluated at *a_x1*.
 ***********************************************************************************************************/

double Branching1d::evaluate( double a_x1 ) const {

    return( m_multiplicity );
}

/*
=========================================================
 * This class is deprecated and should not be being used.
 *
 * @param a_node            [in]    The **HAPI::Node** to be parsed and used to construct the XYs2d.
 * @param a_setupInfo       [in]    Information create my the Protare constructor to help in parsing.
 * @param a_parent          [in]    The parent GIDI::Suite.
 *
 * @return
 */
Branching1dPids::Branching1dPids( HAPI::Node const &a_node, SetupInfo &a_setupInfo, Suite *a_parent ) :
        Form( a_node, a_setupInfo, FormType::branching1dPids, a_parent ),
        m_initial( a_node.attribute_as_string( GIDI_initialChars ) ),
        m_final( a_node.attribute_as_string( GIDI_finalChars ) ) {

}
/*
=========================================================
*/
Branching1dPids::~Branching1dPids( ) {

}

}               // End namespace Functions.

}               // End namespace GIDI.
