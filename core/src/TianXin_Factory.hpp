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
// ***********************************************************************
// @HEADER

#ifndef _TIANXIN_FACTORY_HPP
#define _TIANXIN_FACTORY_HPP

#include <unordered_map>
#include <functional>

namespace TianXin {


    template
    <
        class AbstractProduct,
        typename IdentifierType,
		typename CreatorParmTList
    >
    class Factory
    {
		typedef std::function<AbstractProduct*(CreatorParmTList)> ProductCreator;

	private:
        std::unordered_map<IdentifierType, ProductCreator> associations_;
		
	private:
		Factory() {}
		
		// forbid copy constructor and assign operation,
		// to make sure Factory is a Singleton
        Factory(const Factory&) = delete;
		Factory & operator = (const Factory &) = delete;
        // forbid move constructor and move assign operation,
		// to make sure Factory is a Singleton
		Factory(Factory &&) = delete;
		Factory & operator = (Factory &&) = delete;
		
	public:
		static Factory& Instance()
		{
			static Factory<AbstractProduct,IdentifierType,CreatorParmTList> singleton;
			return singleton;
		}

        bool Register(const IdentifierType& id, ProductCreator creator)
        {
            return associations_.insert(std::make_pair(id, creator)).second != 0;
        }
		
		template<typename Derived>
		bool Register(const IdentifierType& id) {
			return this->Register(id, DerivedCreator<Derived>);
		}

        bool Unregister(const IdentifierType& id)
        {
            return associations_.erase(id) != 0;
        }

        std::unique_ptr<AbstractProduct> Create(const IdentifierType& id, const CreatorParmTList& param)
        {
			auto iter = associations_.find(id);
			if (iter == associations_.end()) return std::unique_ptr<AbstractProduct>(nullptr);
			return std::unique_ptr<AbstractProduct>((iter->second)(param));
        }
		
	private:
		template<typename Derived>
		static AbstractProduct* DerivedCreator(const CreatorParmTList& param) {
			return static_cast<AbstractProduct*>(new Derived(param));
		}

    };
	
}


#endif
