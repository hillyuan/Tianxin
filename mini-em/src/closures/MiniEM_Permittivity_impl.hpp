#ifndef MINIEM_PERMITTIVITY_IMPL_HPP
#define MINIEM_PERMITTIVITY_IMPL_HPP

#include <cmath>

#include "Panzer_BasisIRLayout.hpp"
#include "Panzer_Workset.hpp"
#include "Panzer_Workset_Utilities.hpp"
#include "Panzer_GatherBasisCoordinates.hpp"

namespace mini_em {

//**********************************************************************
template <typename EvalT,typename Traits>
Permittivity<EvalT,Traits>::Permittivity(const std::string & name,
                                         const panzer::IntegrationRule & ir,
                                         const panzer::FieldLayoutLibrary & fl,
                                         const double & epsilon_,
                                         const std::string& DoF_)
{
  using Teuchos::RCP;

  Teuchos::RCP<PHX::DataLayout> data_layout = ir.dl_scalar;
  ir_degree = ir.cubature_degree;

  permittivity = PHX::MDField<ScalarT,Cell,Point>(name, data_layout);
  this->addEvaluatedField(permittivity);
  
  epsilon = epsilon_;

  Teuchos::RCP<const panzer::PureBasis> basis = fl.lookupBasis(DoF_);
  const std::string coordName = panzer::GatherBasisCoordinates<EvalT,Traits>::fieldName(basis->name());
  coords = PHX::MDField<const ScalarT,Cell,Point,Dim>(coordName, basis->coordinates); 
  this->addDependentField(coords);

  std::string n = "Permittivity";
  this->setName(n);
}

//**********************************************************************
template <typename EvalT,typename Traits>
void Permittivity<EvalT,Traits>::evaluateFields(typename Traits::EvalData workset)
{ 
  using panzer::index_t;
  Kokkos::deep_copy(permittivity.get_static_view(), epsilon);

}

//**********************************************************************
}

#endif
