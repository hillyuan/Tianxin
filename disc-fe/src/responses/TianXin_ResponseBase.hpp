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

#ifndef __TianXin_ResponseBase_hpp__
#define __TianXin_ResponseBase_hpp__

#include <string>

#include "Phalanx_config.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_MDField.hpp"
#include "Phalanx_DataLayout_MDALayout.hpp"

#include "Teuchos_RCP.hpp"

#include "Thyra_VectorSpaceBase.hpp"
#include "Thyra_VectorBase.hpp"
#include "Thyra_MultiVectorBase.hpp"
#include "Thyra_LinearOpBase.hpp"

#include "TianXin_WorksetFunctor.hpp"
#include "TianXin_Factory.hpp"

namespace TianXin {
	
/* This class define Dirichlet boundary conditions */
//template<typename EvalT, typename Traits> class ResponseBase;

// **************************************************************
// Residual
// **************************************************************
template<typename EvalT, typename Traits>
class ResponseBase : public PHX::EvaluatorWithBaseImpl<Traits> {
public:
   ResponseBase(const Teuchos::ParameterList& p) {}
	 
   virtual void evaluateFields(typename Traits::EvalData d)=0;
	 
	 
   /** Get the unmodified name for this response.
     */
   std::string getResponseName() const
	{
		return response_name;
	}
	
   // This is the Thyra view of the world
   ///////////////////////////////////////////////////////////
   
   //! Get the vector space for this response, vector space is constructed lazily.
   Teuchos::RCP<const Thyra::VectorSpaceBase<double> > getVectorSpace() const
   { return vSpace_; }

   //! set the vector space for this response
   void setVectorSpace(Teuchos::RCP<const Thyra::VectorSpaceBase<double> > vs)
   { vSpace_ = vs; }

   //! Access the response vector
   Teuchos::RCP<Thyra::VectorBase<double> > getVector() const
   { return tVector_; }

   /** Set the vector (to be filled) for this response. This must be
     * constructed from the vector space returned by <code>getVectorSpace</code>.
     */
   void setVector(const Teuchos::RCP<Thyra::VectorBase<double> > & destVec)
   { tVector_ = destVec; }

protected:
   std::string response_name;
   mutable Teuchos::RCP<const Thyra::VectorSpaceBase<double> > vSpace_;
   Teuchos::RCP<Thyra::VectorBase<double> > tVector_;
};

typedef Factory<ResponseBase<panzer::Traits::Residual,panzer::Traits>,std::string,Teuchos::ParameterList> ResponseResidualFactory;
typedef Factory<ResponseBase<panzer::Traits::Tangent,panzer::Traits>,std::string,Teuchos::ParameterList> ResponseTangentFactory;

}

#endif
