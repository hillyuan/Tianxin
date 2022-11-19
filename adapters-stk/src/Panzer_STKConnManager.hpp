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

#ifndef __Panzer_STKConnManager_hpp__
#define __Panzer_STKConnManager_hpp__

#include <vector>

// Teuchos includes
#include "Teuchos_RCP.hpp"

// Kokkos includes
#include "Kokkos_DynRankView.hpp"

// Panzer includes
#include "Panzer_ConnManager.hpp"

#include "Panzer_STK_Interface.hpp"
#include "Panzer_IntrepidFieldPattern.hpp"

namespace panzer_stk {

class STKConnManager : public panzer::ConnManager {
public:
   typedef typename panzer::ConnManager::LocalOrdinal LocalOrdinal;
   typedef typename panzer::ConnManager::GlobalOrdinal GlobalOrdinal;
   typedef typename Kokkos::DynRankView<GlobalOrdinal,PHX::Device>::HostMirror GlobalOrdinalView;
   typedef typename Kokkos::DynRankView<LocalOrdinal, PHX::Device>::HostMirror LocalOrdinalView;

   STKConnManager(const Teuchos::RCP<const STK_Interface> & stkMeshDB);

   virtual ~STKConnManager() {}

   /** Tell the connection manager to build the connectivity assuming
     * a particular field pattern.
     *
     * \param[in] fp Field pattern to build connectivity for
     */
   virtual void buildConnectivity(const panzer::FieldPattern & fp);

   /** Build a clone of this connection manager, without any assumptions
     * about the required connectivity (e.g. <code>buildConnectivity</code>
     * has never been called).
     */
   virtual Teuchos::RCP<panzer::ConnManager> noConnectivityClone() const;

   /** Get ID connectivity for a particular element
     *
     * \param[in] localElmtId Local element ID
     *
     * \returns Pointer to beginning of indices, with total size
     *          equal to <code>getConnectivitySize(localElmtId)</code>
     */
   virtual const panzer::GlobalOrdinal * getConnectivity(LocalOrdinal localElmtId) const
   { return &connectivity_[elmtLidToConn_[localElmtId]]; }

   /** Get ID connectivity for a particular element
     *
     * \param[in] localElmtId Local element ID
     *
     * \returns Pointer to beginning of indices, with total size
     *          equal to <code>getConnectivitySize(localElmtId)</code>
     */
   virtual panzer::GlobalOrdinal * getConnectivity(LocalOrdinal localElmtId)
   { return &connectivity_[elmtLidToConn_[localElmtId]]; }

   /** How many mesh IDs are associated with this element?
     *
     * \param[in] localElmtId Local element ID
     *
     * \returns Number of mesh IDs that are associated with this element.
     */
   virtual LocalOrdinal getConnectivitySize(LocalOrdinal localElmtId) const
   { return connSize_[localElmtId]; }
   
   const GlobalOrdinalView getConnectivityView()
   { return GlobalOrdinalView(connectivity_.data(), connectivity_.size()); }

   const LocalOrdinalView getConnectivitySizeView()
   { return LocalOrdinalView(connSize_.data(), connSize_.size()); }

   const LocalOrdinalView getElementLidToConnView()
   { return LocalOrdinalView(elmtLidToConn_.data(), elmtLidToConn_.size()); }

   /** How many element blocks in this mesh?
     */
   virtual std::size_t numElementBlocks() const
   { return stkMeshDB_->getNumElementBlocks(); }

   /** Get block IDs from STK mesh object
     */
   virtual void getElementBlockIds(std::vector<std::string> & elementBlockIds) const
   { return stkMeshDB_->getElementBlockNames(elementBlockIds); }
   /** What are the cellTopologies linked to element blocks in this connection manager?
    */
   virtual void getElementBlockTopologies(std::vector<shards::CellTopology> & elementBlockTopologies) const{
     std::vector<std::string> elementBlockIds;
     getElementBlockIds(elementBlockIds);
     elementBlockTopologies.reserve(elementBlockIds.size());
     for (unsigned i=0; i<elementBlockIds.size(); ++i) {
       elementBlockTopologies.push_back(*(stkMeshDB_->getCellTopology(elementBlockIds[i])));
     }
   }
   /** Get the local element IDs for a paricular element
     * block. These are only the owned element ids.
     *
     * \param[in] blockIndex Block Index
     *
     * \returns Vector of local element IDs.
     */
   virtual const std::vector<LocalOrdinal> & getElementBlock(const std::string & blockId) const
   { return *(elementBlocks_.find(blockId)->second); }
   
   /** Get the local element IDs for a paricular element block. 
     *
     * \param[in] blockIndex Block Index
     *
     * \returns Vector of local element IDs.
     */
   virtual void getElementBlockAll(const std::string & blockId, std::vector<LocalOrdinal>& elementLid) const final
   {
	/*    auto& owned = *(elementBlocks_.find(blockId)->second);
		auto& ghosted = *(neighborElementBlocks_.find(blockId)->second);
		std::cout << owned.size() << ", " << ghosted.size() << " ********\n";
		owned.insert(owned.end(), ghosted.begin(), ghosted.end());std::cout << owned.size() << " ********\n";
		return  owned;*/
		stkMeshDB_ -> getAllElementIDs(blockId, elementLid);
	}

   /** Get the local element IDs for a paricular element
     * block. These element ids are not owned, and the element
     * will live on another processor.
     *
     * \param[in] blockIndex Block Index
     *
     * \returns Vector of local element IDs.
     */
   virtual const std::vector<LocalOrdinal> & getNeighborElementBlock(const std::string & blockId) const
   { return *(neighborElementBlocks_.find(blockId)->second); }

   /** Get the coordinates (with local cell ids) for a specified element block and field pattern.
     *
     * \param[in] blockId Block containing the cells
     * \param[in] coordProvider Field pattern that builds the coordinates
     * \param[out] localCellIds Local cell Ids (indices)
     * \param[out] Resizable field container that contains the coordinates
     *             of the points on exit.
     */
   virtual void getDofCoords(const std::string & blockId,
                             const panzer::Intrepid2FieldPattern & coordProvider,
                             std::vector<std::size_t> & localCellIds,
                             Kokkos::DynRankView<double,PHX::Device> & points) const;

    /** Get STK interface that this connection manager is built on.
      */
    Teuchos::RCP<const STK_Interface> getSTKInterface() const
    { return stkMeshDB_; }

    /** How many elements are owned by this processor. Further,
      * the ordering of the local ids is suct that the first
      * <code>getOwnedElementCount()</code> elements are owned
      * by this processor. This is true only because of the
      * local element ids generated by the <code>STK_Interface</code>
      * object.
      */
    std::size_t getOwnedElementCount() const
    { return ownedElementCount_; }

    /** Before calling buildConnectivity, provide sideset IDs from which to
      * extract associated elements.
      */
    void associateElementsInSideset(const std::string sideset_id);

    /** After calling <code>buildConnectivity</code>, optionally check which
      * sidesets yielded no element associations in this communicator. This is a
      * parallel operation. In many applications, the outcome indicating
      * correctness is that the returned vector is empty.
      */
    std::vector<std::string> checkAssociateElementsInSidesets(const Teuchos::Comm<int>& comm) const;

    /** Get elements, if any, associated with <code>el</code>, excluding
      * <code>el</code> itself.
      */
    virtual const std::vector<LocalOrdinal>& getAssociatedNeighbors(const LocalOrdinal& el) const;

    /** Return whether getAssociatedNeighbors will return true for at least one
      * input. Default implementation returns false.
      */
    virtual bool hasAssociatedNeighbors() const;
	
	/** Get the node connectivity of a given element
     *
     * \param[in] elmtLid elemental local index
     *
     * \param[out] nodesgid Vector of global nodes IDs.
     */
    void getElementalNodeConnectivity(const LocalOrdinal& elmtLid, std::vector<GlobalOrdinal>& nodesgid) const;
    int getNodeRank() const final {return stkMeshDB_->getNodeRank();}
	
    /** Get the node connectivity of a given element
     *
     * \param[in] elmtLid elemental local index
     *
     * \param[out] nodesgid Vector of global edges IDs.
     */
    void getElementalEdges(const LocalOrdinal& elmtLid, std::vector<GlobalOrdinal>& nodesgid) const;
    int getEdgeRank() const final {return stkMeshDB_->getEdgeRank();}
	
	/** Get the node connectivity of a given element
     *
     * \param[in] elmtLid elemental local index
     *
     * \param[out] nodesgid Vector of global faces IDs.
     */
    void getElementalFaces(const LocalOrdinal& elmtLid, std::vector<GlobalOrdinal>& nodesgid) const;
    int getFaceRank() const final {return stkMeshDB_->getFaceRank();}

    int getElementRank() const final {return stkMeshDB_->getElementRank();}

	virtual void fillLocalCellIDs(Kokkos::View<panzer::GlobalOrdinal*> & owned_cells,
                 Kokkos::View<panzer::GlobalOrdinal*> & ghost_cells,
                 Kokkos::View<panzer::GlobalOrdinal*> & virtual_cells) final
	{
		stkMeshDB_->fillLocalCellIDs(owned_cells,ghost_cells,virtual_cells);
	}

protected:
   /** Apply periodic boundary conditions associated with the mesh object.
     *
     * \note This function requires global All-2-All communication IFF
     *       periodic boundary conditions are required.
     */
   void applyPeriodicBCs( const panzer::FieldPattern & fp, GlobalOrdinal nodeOffset, GlobalOrdinal edgeOffset,
                                                           GlobalOrdinal faceOffset, GlobalOrdinal cellOffset);
   void applyInterfaceConditions();

   void buildLocalElementMapping();
   void clearLocalElementMapping();
   void buildOffsetsAndIdCounts(const panzer::FieldPattern & fp,
                                LocalOrdinal & nodeIdCnt, LocalOrdinal & edgeIdCnt,
                                LocalOrdinal & faceIdCnt, LocalOrdinal & cellIdCnt,
                                GlobalOrdinal & nodeOffset, GlobalOrdinal & edgeOffset,
                                GlobalOrdinal & faceOffset, GlobalOrdinal & cellOffset) const;

   LocalOrdinal addSubcellConnectivities(stk::mesh::Entity element,unsigned subcellRank,
                                         LocalOrdinal idCnt,GlobalOrdinal offset);

   void modifySubcellConnectivities(const panzer::FieldPattern & fp, stk::mesh::Entity element,
                                    unsigned subcellRank,unsigned subcellId,GlobalOrdinal newId,GlobalOrdinal offset);

   Teuchos::RCP<const STK_Interface> stkMeshDB_;

   std::vector<stk::mesh::Entity> elements_;

   // element block information
   std::map<std::string,Teuchos::RCP<std::vector<LocalOrdinal> > > elementBlocks_;
   std::map<std::string,Teuchos::RCP<std::vector<LocalOrdinal> > > neighborElementBlocks_;
   std::map<std::string,GlobalOrdinal> blockIdToIndex_;

   std::vector<LocalOrdinal> elmtLidToConn_; // element LID to Connectivity map
   std::vector<LocalOrdinal> connSize_; // element LID to Connectivity map
   std::vector<GlobalOrdinal> connectivity_; // Connectivity

   std::size_t ownedElementCount_;

   std::vector<std::string> sidesetsToAssociate_;
   std::vector<bool> sidesetYieldedAssociations_;
   std::vector<std::vector<LocalOrdinal> > elmtToAssociatedElmts_;
};

}

#endif
