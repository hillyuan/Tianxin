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

#ifndef _POINT_EVALUATOR_HPP
#define _POINT_EVALUATOR_HPP

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Teuchos_ParameterList.hpp"

#include "TianXin_WorksetFunctor.hpp"
#include "TianXin_AbstractDiscretation.hpp"
#include "Panzer_GlobalIndexer.hpp"

#include <map>
#include <cassert>

namespace TianXin {

// **************************************************************
// This class defines evaluators act upon DOF points specified by (sideset, dofname) pair
// **************************************************************

template<typename EvalT, typename Traits>
class PointEvaluatorBase : public PHX::EvaluatorWithBaseImpl<Traits>
{
  private:
    typedef typename Traits::RealType RealType;
    typedef typename EvalT::ScalarT ScalarT;
	typedef typename panzer::LocalOrdinal LO;
	typedef typename panzer::GlobalOrdinal GO;

  private:
    int                          m_group_id;         // group id maybe used in activation
	int                          m_sideset_rank;     // 0: node; 1: edge; 2: face; 3: volume
    std::string                  m_sideset_name;     // sideset this Dirichlet condition act upon
    Teuchos::Array<std::string>  m_dof_name;         // ux,uy,uz etc
    std::string                  m_value_type;       // evaluator name

  public:
    PointEvaluatorBase(const Teuchos::ParameterList& params, const Teuchos::RCP<const TianXin::AbstractDiscretation>&,
      const Teuchos::RCP<const panzer::GlobalIndexer> & indexer );

    int getGroupID() const
	{ return m_group_id; }

    const std::string getSideSetName() const
    { return m_sideset_name; }

    const std::string getDofName(int i) const
    {	
		assert(i>0 && i<m_dof_name.size());
		return m_dof_name[i]; 
	}
	
	void preEvaluate(typename Traits::PreEvalData d);
	//void postRegistrationSetup(typename Traits::SetupData d, PHX::FieldManager<Traits>& vm);

    // This function will be overloaded with template specialized code
    void evaluateFields(typename Traits::EvalData d)=0;

  protected:
    double m_penalty;
	std::unique_ptr<TianXin::WorksetFunctor> m_pFunctor;
	//working variables
	std::size_t m_ndofs;
    Kokkos::View<panzer::LocalOrdinal*, Kokkos::HostSpace>   m_local_dofs;
    Kokkos::View<panzer::GlobalOrdinal*, Kokkos::HostSpace>  m_global_dofs;
	Kokkos::View<RealType*, Kokkos::HostSpace>               m_values;
	Teuchos::RCP<panzer::LinearObjContainer>  m_GhostedContainer; 
    //Teuchos::RCP<Xpetra::CrsMatrix<ScalarT, LO, GO, KokkosClassic::DefaultNode::DefaultNodeType> >  m_crsmatrix;
	void setValues(const panzer::Workset&);
};

}

#include "TianXin_PointEvaluator_impl.hpp"

#endif
