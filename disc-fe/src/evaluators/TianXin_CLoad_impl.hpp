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

#ifndef _TIANXIN_CLOAD_DIRICHLET_IMPL_HPP
#define _TIANXIN_CLOAD_DIRICHLET_IMPL_HPP

#include <set>
#include <stdexcept>

namespace TianXin {

// **************************************************************
// Residual
// **************************************************************

template<typename Traits>
CLoadEvalautor<panzer::Traits::Residual,Traits>::CLoadEvalautor(const Teuchos::ParameterList& params, const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const Teuchos::RCP<const panzer::GlobalIndexer> & indexer )
: PointEvaluatorBase<panzer::Traits::Residual,Traits>(params, mesh, indexer )
{}

template<typename Traits>
void CLoadEvalautor<panzer::Traits::Residual, Traits> :: evaluateFields(typename Traits::EvalData d)
{
	this->setValues(d);
	this->m_GhostedContainer->applyConcentratedLoad(this->m_local_dofs, this->m_values);
}

// **************************************************************
// Jacobian
// **************************************************************

template<typename Traits>
CLoadEvalautor<panzer::Traits::Jacobian,Traits>::CLoadEvalautor(const Teuchos::ParameterList& params, const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const Teuchos::RCP<const panzer::GlobalIndexer> & indexer )
: PointEvaluatorBase<panzer::Traits::Jacobian,Traits>(params, mesh, indexer )
{}

template<typename Traits>
void CLoadEvalautor<panzer::Traits::Jacobian, Traits> :: evaluateFields(typename Traits::EvalData d)
{
	// Do something here when DOF-dependent concerntrated load
}

// **************************************************************
// Tangent
// **************************************************************

template<typename Traits>
CLoadEvalautor<panzer::Traits::Tangent,Traits>::CLoadEvalautor(const Teuchos::ParameterList& params, const Teuchos::RCP<const TianXin::AbstractDiscretation>& mesh,
      const Teuchos::RCP<const panzer::GlobalIndexer> & indexer )
: PointEvaluatorBase<panzer::Traits::Tangent,Traits>(params, mesh, indexer )
{}

template<typename Traits>
void CLoadEvalautor<panzer::Traits::Tangent, Traits> :: evaluateFields(typename Traits::EvalData d)
{}

}

#endif
