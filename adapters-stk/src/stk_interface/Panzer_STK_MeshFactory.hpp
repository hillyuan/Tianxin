// @HEADER
// ***********************************************************************
//
//           Panzer: A partial differential equation assembly
//       engine for strongly coupled complex multiphysics systems
//                 Copyright (2011) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Roger P. Pawlowski (rppawlo@sandia.gov) and
// Eric C. Cyr (eccyr@sandia.gov)
// ***********************************************************************
// @HEADER

#ifndef Panzer_STK_MeshFactory_hpp__
#define Panzer_STK_MeshFactory_hpp__

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterListAcceptorDefaultBase.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include "Panzer_STK_PeriodicBC_Parser.hpp"

namespace panzer_stk {

class STK_Interface;

/** Pure virtual interface that constructs a 
  * STK_Mesh interface object.
  */
class STK_MeshFactory : public Teuchos::ParameterListAcceptorDefaultBase {
public:
   STK_MeshFactory() : enableRebalance_(false) {}

   /** Construct a STK_Inteface object described
     * by this factory.
     *
     * \param[in] parallelMach Descriptor for machine to build this mesh on.
     *
     * \returns Pointer to <code>STK_Interface</code> object with 
     *          <code>isModifiable()==false</code>.
     */ 
   virtual Teuchos::RCP<STK_Interface> buildMesh(stk::ParallelMachine parallelMach) const = 0;

   /** This builds all the meta data of the mesh. Does not call metaData->commit.
     * Allows user to add solution fields and other pieces. The mesh can be "completed"
     * by calling <code>completeMeshConstruction</code>.
     */
   virtual Teuchos::RCP<STK_Interface> buildUncommitedMesh(stk::ParallelMachine parallelMach) const = 0;

   /** Finishes building a mesh object started by <code>buildUncommitedMesh</code>.
     */
   virtual void completeMeshConstruction(STK_Interface & mesh,stk::ParallelMachine parallelMach) const = 0;

   /** Parse the periodic boundary condition parameter list and build a vector of periodic boundary
     * conditions (a convenience function)
     */
   static void parsePeriodicBCList(const Teuchos::RCP<Teuchos::ParameterList> & pl,
                                   std::vector<Teuchos::RCP<const PeriodicBC_MatcherBase> > & periodicBC)
   {
      panzer_stk::PeriodicBC_Parser parser;
      parser.setParameterList(pl);
      periodicBC = parser.getMatchers();
   }
   
   static void parsePeriodicBCList(const Teuchos::ParameterList & pl,
             std::vector< std::tuple<std::string, std::string, std::string> > & periodicBC)
   {
	  std::string type, s0, s1;
	  periodicBC.clear();
	   
      int numBCs = pl.get<int>("Count");
	  for(int i=1;i<=numBCs;i++) {
		std::stringstream ss;
		ss << "Periodic Condition " << i; 
        std::string cond = pl.get<std::string>(ss.str());
        std::string::size_type d0 = cond.find_first_of(' ');
        std::string matcher = cond.substr(0,d0);
        if( matcher.find("coord") ) {
           type = std::string("NodeSet");
        } else if( matcher.find("edge") ) {
           type = std::string("EdgeSet");
		} else if( matcher.find("face") ) {
           type = std::string("FaceSet");
		} else if( matcher.find("all") ) {
           type = std::string("All");
		} else {
			type = std::string("");
		};
		std::string::size_type d1 = cond.find_first_of(';');
        s0 = cond.substr(d0+1,d1-d0-1);
		s1 = cond.substr(d1+1,cond.length()-d1-1);
		std::tuple<std::string, std::string, std::string> t = std::make_tuple(type, s0, s1);
		periodicBC.emplace_back( t );
	  }
   }

   void enableRebalance(bool enable,const Teuchos::RCP<const Teuchos::ParameterList> & rebalanceList=Teuchos::null) 
   { enableRebalance_ = enable; 
     rebalanceList_ = rebalanceList; }

   void rebalance(STK_Interface & mesh) const
   {
     if(rebalanceList_!=Teuchos::null) {
       // loop over user specified partitioning lists
       for(Teuchos::ParameterList::ConstIterator itr=rebalanceList_->begin();
           itr!=rebalanceList_->end();++itr) {

         const Teuchos::ParameterEntry & entry = rebalanceList_->entry(itr);
         TEUCHOS_TEST_FOR_EXCEPTION(!entry.isList(),std::runtime_error,
                                    "Rebalance list is incorrect:\n" << entry << "\nA Zoltan list formated with strings is expected.");

         // partition according to the list
         mesh.rebalance(Teuchos::getValue<Teuchos::ParameterList>(entry));

         // rebuild mesh internals
         mesh.buildLocalElementIDs();
       }
     }
     else if(enableRebalance_) {
       // do the default thing, once
       Teuchos::ParameterList emptyList;
       mesh.rebalance(emptyList);

       // rebuild mesh internals
       mesh.buildLocalElementIDs();
     }
   }

   double getMeshCoord(const int nx, const double deltaX, const double x0) const {
      double x = static_cast<double>(nx)*deltaX;
      double modX = std::abs(x);
      double modX0 = std::abs(x0);
      double val = x+x0;
      if ((x0*x < 0.0) && (std::abs(modX-modX0) < std::numeric_limits<double>::epsilon()*modX0)) val=0.0;
      return (val);
   }

protected:
   // vector of periodic boundary condition objects
   std::vector<Teuchos::RCP<const PeriodicBC_MatcherBase> > periodicBCVec_; 
   std::vector< std::tuple<std::string, std::string, std::string> > periodicity_;

   // for managing rebalance
   bool enableRebalance_;
   Teuchos::RCP<const Teuchos::ParameterList> rebalanceList_;
};

}

#endif
