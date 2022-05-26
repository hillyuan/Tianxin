// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) Xi Yuan
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
// @HEADER

#ifndef _BC_DIRICHLET_HPP
#define _BC_DIRICHLET_HPP

#include <map>
##include <cassert>

namespace TianXin {

enum class DiricheltStrategy : int
{
	1_0, Penalty, Lagrarangian
};

//**********************************************************************
template< typename ScalarT >
class Dirichlet
{
  private:
    integer                      m_group_id;         // group id maybe used in activation
	DiricheltStrategy            m_strategy;         // algortithm to deal with dirichlet consition
	int                          m_sideset_rank;     // 0: node; 1: edge; 2: face; 3: volume
    std::string                  m_sideset_name;     // sideset this Dirichlet condition act upon
    Teuchos::Array<std::string>  m_dof_name;         // ux,uy,uz etc
    std::string                  m_value;            // evaluator name

  public:
    Dirichlet(const Teuchos::ParameterList& params );

    std::string getType() const
    { return m_type; }

    const std::string getSideSetName() const
    { return m_sideset_name; }

    const std::string getDofName(int i) const
    {	
		assert(i>0 && i<m_dof_name.size());
		return m_dof_name[i]; 
	}

  public:
    double m_penalty;
    Teuchos::RCP<Teuchos::ParameterList> m_params;
};

}

#include "TianXin_Dirichlet_impl.hpp"

#endif
