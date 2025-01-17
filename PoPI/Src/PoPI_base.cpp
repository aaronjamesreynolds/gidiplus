/*
# <<BEGIN-copyright>>
# Copyright 2019, Lawrence Livermore National Security, LLC.
# See the top-level COPYRIGHT file for details.
# 
# SPDX-License-Identifier: MIT
# <<END-copyright>>
*/

#include "PoPI.hpp"

#define PoPI_idChars "id"
#define PoPI_symbolChars "symbol"

namespace PoPI {

/*
=========================================================
========================== Base =========================
=========================================================
*/
Base::Base( std::string const &a_id, Particle_class a_class ) :
        m_id( a_id ),
        m_class( a_class ) {

}
/*
=========================================================
*/
Base::Base( HAPI::Node const &a_node, std::string const &a_label, Particle_class a_class ) :
        m_id( a_node.attribute( a_label.c_str( ) ).value( ) ),
        m_class( a_class ),
        m_index( -1 ) {

}
/*
=========================================================
*/
Base::~Base( ) {

}

/*
=========================================================
========================= IDBase ========================
=========================================================
*/
IDBase::IDBase( std::string const &a_id, Particle_class a_class ) :
        Base( a_id, a_class ) {

}
/*
=========================================================
*/
IDBase::IDBase( HAPI::Node const &a_node, Particle_class a_class ) :
        Base( a_node, PoPI_idChars, a_class ) {

}
/*
=========================================================
*/
IDBase::~IDBase( ) {

}
/*
=========================================================
*/
int IDBase::addToDatabase( Database *a_DB ) {

    a_DB->add( this );
    return( index( ) );
}

/*
============================================================
======================== SymbolBase ========================
============================================================
*/
SymbolBase::SymbolBase( HAPI::Node const &a_node, Particle_class a_class ) :
        Base( a_node, PoPI_symbolChars, a_class ) {

}
/*
=========================================================
*/
SymbolBase::~SymbolBase( ) {

}
/*
=========================================================
*/
int SymbolBase::addToSymbols( Database *a_DB ) {

    a_DB->addSymbol( this );
    return( index( ) );
}

}
