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

#ifndef __Panzer_BlockedDOFManagerFactory_decl_hpp__
#define __Panzer_BlockedDOFManagerFactory_decl_hpp__

#include "Panzer_GlobalIndexerFactory.hpp"

namespace panzer {

class BlockedDOFManagerFactory : public virtual GlobalIndexerFactory {
public:
   BlockedDOFManagerFactory() : useTieBreak_(false) {}
   virtual ~BlockedDOFManagerFactory() {}

   /** Does a fieldOrder string require blocking? 
     * A field order is basically stetup like this
     *    blocked: <field 0> <field 1> 
     * where two blocks will be created. To merge fields
     * between blocks use a hyphen, i.e.
     *    blocked: <field 0> <field 1> - <field 2> - <field 3>
     * This will create 2 blocks, the first contains only <field 0>
     * and the second combines <field 1>, <field 2> and <field 3>. Note
     * the spaces before and after the hyphen, these are important!
     */
   static bool requiresBlocking(const std::string & fieldorder);

   /** Does a fieldOrder string require blocking? 
     * A field order is basically stetup like this
     *    blocked: <field 0> <field 1> 
     * where two blocks will be created. To merge fields
     * between blocks use a hyphen, i.e.
     *    blocked: <field 0> <field 1> - <field 2> - <field 3>
     * This will create 2 blocks, the first contains only <field 0>
     * and the second combines <field 1>, <field 2> and <field 3>. Note
     * the spaces before and after the hyphen, these are important!
     */
   static void buildBlocking(const std::string & fieldorder,std::vector<std::vector<std::string> > & blocks);

   /** Use the physics block to construct a unique global indexer object.
     * 
     * \param[in] mpiComm MPI communicator to use in the construction
     * \param[in] physicsBlocks A vector of physics block objects that contain
     *                          unknown field information.
     * \param[in] connMngr Connection manager that contains the mesh topology
     * \param[in] fieldOrder Specifies the local ordering of the degrees of
     *            freedom. This is relevant when degrees of freedom are shared
     *            on the same geometric entity. The default is an alphabetical
     *            ordering.
     * \param[in] buildGlobalUnknowns Build the global unknowns before
     *            returning. The default value gives backwards-compatible
     *            behavior. Set this to false if the caller will initialize the
     *            DOF manager in additional ways before issuing the call to
     *            build the global unknowns itself.
     *
     * \returns A GlobalIndexer object. If buildGlobalUnknowns is true,
     *          the object is fully constructed. If it is false, the caller must
     *          finalize it.
     */
   virtual Teuchos::RCP<panzer::GlobalIndexer> 
   buildGlobalIndexer(const Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> > & mpiComm,
                            const std::vector<Teuchos::RCP<panzer::PhysicsBlock> > & physicsBlocks,
                            const Teuchos::RCP<ConnManager> & connMngr,
                            const std::string & fieldOrder="") const;

   void setUseTieBreak(bool flag) 
   { useTieBreak_ = flag; }

   bool getUseTieBreak()
   { return useTieBreak_; }

private:
   bool useTieBreak_;
};

}

#endif
