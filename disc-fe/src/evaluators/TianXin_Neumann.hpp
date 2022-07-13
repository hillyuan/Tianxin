// @HEADER
// ***********************************************************************
//
//           TianXin: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2022) YUAN Xi
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

#ifndef _TIANXIN_NEUMANN_EVALUATOR_HPP
#define _TIANXIN_NEUMANN_EVALUATOR_HPP

#include "Phalanx_Evaluator_Macros.hpp"
#include "Phalanx_Evaluator_Factory.hpp"
#include "Phalanx_MDField.hpp"

#include "TianXin_WorksetFunctor.hpp"
#include "TianXin_Factory.hpp"
#include "Panzer_LinearObjFactory.hpp"

#include <memory>

namespace TianXin {
    
  /** \brief Evaluates a Neumann BC residual contribution

      computes the surface integral term resulting from integration
      by parts for a particular dof:

      int(n \cdot (flux * phi) )
  */
template<typename EvalT, typename Traits>
class NeumannBase : public PHX::EvaluatorWithBaseImpl<Traits>
{
    using ScalarT = typename EvalT::ScalarT;

public:

    NeumannBase(const Teuchos::ParameterList& p);

    void postRegistrationSetup(typename Traits::SetupData d,PHX::FieldManager<Traits>& fm);
    virtual void evaluateFields(typename Traits::EvalData d) = 0;
	
	Teuchos::RCP< PHX::Evaluator<Traits> > buildScatter(const Teuchos::ParameterList& p,
		const panzer::LinearObjFactory<Traits>& lof);

protected:
    // output  
    PHX::MDField<ScalarT> residual;

    // common data used by neumann calculation
	std::string residual_name;
	std::string dof_name;
    std::string basis_name;
    std::size_t basis_index, ir_index;
	
	std::unique_ptr<TianXin::WorksetFunctor> pFunc;
	int quad_order, quad_index;
	std::size_t num_cell, num_qp, num_dim;
	PHX::MDField<ScalarT,panzer::Cell,panzer::Point,panzer::Dim> normals;
	//Kokkos::DynRankView<ScalarT, PHX::Device> normal_lengths_buffer;
	void calculateNormal(typename Traits::EvalData d);
public:
  // for testing purposes
  const PHX::FieldTag & getFieldTag() const 
  { return residual.fieldTag(); }
}; // end of class NeumannBase

typedef Factory<NeumannBase<panzer::Traits::Residual,panzer::Traits>,std::string,Teuchos::ParameterList> NeumannResidualFactory;
typedef Factory<NeumannBase<panzer::Traits::Jacobian,panzer::Traits>,std::string,Teuchos::ParameterList> NeumannJacobianFactory;


// **************************************************************
// Scalar Flux, e.g. Heat Flux dT/dn
// **************************************************************

template<typename EvalT, typename Traits>
class Flux : public NeumannBase<EvalT, Traits>
{
  using ScalarT = typename EvalT::ScalarT;
  public:
    Flux(const Teuchos::ParameterList& params );
    void evaluateFields(typename Traits::EvalData d) final;
};
namespace NeumannRegister {
	static bool const FLUX_ROK = NeumannResidualFactory::Instance().template Register< Flux<panzer::Traits::Residual,panzer::Traits> >( "Flux");
	static bool const FLUX_JOK = NeumannJacobianFactory::Instance().template Register< Flux<panzer::Traits::Jacobian,panzer::Traits> >( "Flux");
}

// **************************************************************
// Pressure act upon surface
// **************************************************************

/*template<typename EvalT, typename Traits>
class Pressure : public NeumannBase<EvalT, Traits>
{
  public:
    Pressure(const Teuchos::ParameterList& params );
    void evaluateFields(typename Traits::EvalData d) final;
  private:
    std::unique_ptr<TianXin::WorksetFunctor> pFunc;
	int quad_order, quad_index;
	std::size_t num_cell,num_qp, num_dim;
	PHX::MDField<ScalarT,Cell,Point,Dim> normals;
};
namespace NeumannRegister {
	static bool const FLUX_ROK = NeumannFunctorFactory::Instance().template Register< Pressure<panzer::Traits::Residual,panzer::Traits> >( "Pressure");
	static bool const FLUX_JOK = NeumannFunctorFactory::Instance().template Register< Pressure<panzer::Traits::Jacobian,panzer::Traits> >( "Pressure");
}
*/

}

#include "TianXin_Neumann_impl.hpp"

#endif
